# -*- code = utf-8 -*-
# @Time: 2022/10/22 13:42
# @Author: Chen Zigeng
# @File:part2_new.py
# @Software:PyCharm
import cv2
import os
import unittest
import numpy as np
#matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
# import gco
from gco import pygco as gco


def distance(c1,c2):
    res = abs(int(c1[0])-int(c2[0]))+abs(int(c1[1])-int(c2[1]))+abs(int(c1[2])-int(c2[2]))
    return res/15

img_left = cv2.imread("images/im6.png")
img_right = cv2.imread("images/im2.png")
print(img_left.shape)
print(img_right.shape)

d_num =65
w,h,c = img_left.shape
image = np.zeros([w,h])
unary = np.tile(image[:, :, np.newaxis], [1, 1, d_num])

lamda = 0.1
smooth = lamda - lamda * np.eye(d_num)

for i in range(d_num):
    for j in range(d_num):
        if i != j:
            smooth[i][j] = abs(i-j)*lamda
print(smooth)

maxi =0
for i in range(w):
    for j in range(h):
        for d in range(d_num):
            if j+d<h:
                unary[i][j][d] = distance(img_left[i][j],img_right[i][j+d])
                maxi = max(unary[i][j][d],maxi)
            # else:
            #     unary[i][j][d] = 100
# max = np.max(np.hstack(unary))
#unary = unary/maxi

#print(unary)
labels = gco.cut_grid_graph_simple(unary, smooth, connect=8,n_iter=-1)

labels = np.reshape(labels, [w, h])
labels = labels
#labels = labels*255/d_num
print("shape",labels.shape)
labels = labels
print(labels)
plt.figure(num=1, dpi=80, figsize=(8,6))
plt.imshow(labels,cmap="gray")
plt.axis("off")
plt.title("Best Result")
plt.show()