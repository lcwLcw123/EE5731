# -*- code = utf-8 -*-
# @Time: 2022/10/20 22:57
# @Author: Chen Zigeng
# @File:part3.py
# @Software:PyCharm

import cv2
import os
import unittest
import numpy as np
#matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
# import gco
from gco import pygco as gco
from scipy import linalg

K1 = np.array([[1221.2270770,0.0000000,479.5000000],[0.0000000,1221.2270770,269.5000000],[0.0000000,0.0000000,1.0000000]])
R1 = np.array([[1.0000000000,0.0000000000,0.0000000000],[0.0000000000,1.0000000000,0.0000000000],[0.0000000000,0.0000000000,1.0000000000]])
T1 = np.array([[0.0000000000],[0.0000000000],[0.0000000000]])
K2 = np.array([[1221.2270770,0.0000000,479.5000000],[0.0000000,1221.2270770,269.5000000],[0.0000000,0.0000000,1.0000000]])
R2 = np.array([[0.9998813487,0.0148994942,0.0039106989],[-0.0148907594,0.9998865876,-0.0022532664],[-0.0039438279,0.0021947658,0.9999898146]])
T2 = np.array([[-9.9909793759],[0.2451742154],[0.1650832670]])

img_left = cv2.imread("part3_1.jpg")
img_right = cv2.imread("part3_2.jpg")

def distance(c1,c2):
    res = abs(int(c1[0])-int(c2[0]))+abs(int(c1[1])-int(c2[1]))+abs(int(c1[2])-int(c2[2]))
    return res/10

def get_part1(k1,r1,k2,r2):
    m = np.dot(k2,r2.T)
    n = np.dot(r1,linalg.inv(k1))
    return np.dot(m,n)

def get_part2(k2,r2,t1,t2):
    m = np.dot(k2,r2.T)
    n = (t1-t2)
    return np.dot(m,n)

print(img_left.shape)
print(img_right.shape)

part1 = get_part1(K1,R1,K2,R2)
part2 = get_part2(K2,R2,T1,T2)
print(part1)
print(part2)
disparity = np.arange(0.0001,0.01,0.0002)
w,h,c = img_left.shape
image = np.zeros([w,h])
unary = np.tile(image[:, :, np.newaxis], [1, 1, len(disparity)])

lamda = 0.1
smooth = lamda - lamda * np.eye(len(disparity))
maxi =0
for i in range(w):
    for j in range(h):
        for d in range(len(disparity)):
            deep = disparity[d]
            coo_old = np.array([[j,i,1]])
            coo_new = np.dot(part1,coo_old.T)+deep*part2
            jnew = int(coo_new[0]/coo_new[2])
            inew = int(coo_new[1]/coo_new[2])
            if inew>=0 and inew<w and jnew>=0 and jnew<h:
                unary[i][j][d] = distance(img_left[i][j], img_right[inew][jnew])
                maxi = max(unary[i][j][d],maxi)

unary = unary/maxi
print(unary.shape)
labels = gco.cut_grid_graph_simple(unary, smooth, connect=4,n_iter=-1)

labels = np.reshape(labels, [w, h])

print("shape",labels.shape)
labels = labels
plt.figure(num=1, dpi=80, figsize=(8,8))
plt.imshow(labels,cmap="gray")
plt.axis("off")
plt.title("Best Result")
plt.show()
