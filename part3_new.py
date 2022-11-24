# -*- code = utf-8 -*-
# @Time: 2022/10/22 13:59
# @Author: Chen Zigeng
# @File:part3_new.py
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



img_left = cv2.imread("images/test00.jpg")
img_right = cv2.imread("images/test09.jpg")

def distance(c1,c2):
    res = abs(int(c1[0])-int(c2[0]))+abs(int(c1[1])-int(c2[1]))+abs(int(c1[2])-int(c2[2]))
    return res/20

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
k1 = np.array([[1221.2270770,0.0000000,479.5000000],[0.0000000,1221.2270770,269.5000000],[0.0000000,0.0000000,1.0000000]])
r1 = np.array([[1.0000000000,0.0000000000,0.0000000000],[0.0000000000,1.0000000000,0.0000000000],[0.0000000000,0.0000000000,1.0000000000]])
t1 = np.array([[0.0000000000],[0.0000000000],[0.0000000000]])
k2 = np.array([[1221.2270770,0.0000000,479.5000000],[0.0000000,1221.2270770,269.5000000],[0.0000000,0.0000000,1.0000000]])
r2 = np.array([[0.9998813487,0.0148994942,0.0039106989],[-0.0148907594,0.9998865876,-0.0022532664],[-0.0039438279,0.0021947658,0.9999898146]])
t2 = np.array([[-9.9909793759],[0.2451742154],[0.1650832670]])
part1 = get_part1(k1,r1,k2,r2)
part2 = get_part2(k2,r2,t1,t2)
print(part1)
print(part2)
drange = np.arange(0,0.01,0.01/65)
w,h,c = img_left.shape
image = np.zeros([w,h])
unary = np.tile(image[:, :, np.newaxis], [1, 1, len(drange)])

lamda = 0.1
smooth = lamda - lamda * np.eye(len(drange))
for i in range(len(drange)):
    for j in range(len(drange)):
        if i != j:
            smooth[i][j] = abs(i-j)*lamda
print(smooth)
maxitem =-1
for i in range(w):
    for j in range(h):
        for d in range(len(drange)):
            deep = drange[d]
            coo_old = np.array([[j,i,1]])
            coo_new = np.dot(part1,coo_old.T)+deep*part2
            jnew = int(coo_new[0]/coo_new[2])
            inew = int(coo_new[1]/coo_new[2])
            if inew>=0 and inew<w and jnew>=0 and jnew<h:
                unary[i][j][d] = distance(img_left[i][j], img_right[inew][jnew])
                maxitem = max(unary[i][j][d],maxitem)
            # else:
            #     unary[i][j][d] = 100
# max = np.max(np.hstack(unary))
#unary = unary/maxitem
print(unary.shape)
labels = gco.cut_grid_graph_simple(unary, smooth, connect=8,n_iter=-1)

labels = np.reshape(labels, [w, h])
#labels = labels*255/d_num
print("shape",labels.shape)
labels = labels
plt.figure(num=1, dpi=80, figsize=(8,8))
plt.imshow(labels,cmap="gray")
plt.axis("off")
plt.title("Best Result")
plt.show()