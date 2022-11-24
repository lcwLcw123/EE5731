# -*- code = utf-8 -*-
# @Time: 2022/10/21 19:43
# @Author: Chen Zigeng
# @File:part4.py
# @Software:PyCharm
import cv2
import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
# import gco
from gco import pygco as gco
from scipy import linalg
from tqdm import tqdm
import os

def distance(c1,c2):
    res = abs(int(c1[0])-int(c2[0]))+abs(int(c1[1])-int(c2[1]))+abs(int(c1[2])-int(c2[2]))
    return res/10
def l2(c1,c2):
    np.square(c1-c2)
    return np.sum(np.square(c1 - c2), axis=-1)
def get_part1(k1,r1,k2,r2):
    m = np.dot(k2,r2.T)
    n = np.dot(r1,linalg.inv(k1))
    return np.dot(m,n)
def get_part2(k2,r2,t1,t2):
    m = np.dot(k2,r2.T)
    n = (t1-t2)
    n = n.T
    return np.dot(m,n)
def get_video(path):
    file_name_list = os.listdir(path)
    return file_name_list
def get_KRT(filepath):
    K = []
    R = []
    T = []
    f = open(filepath)
    frame_n = int(f.readline())
    f.readline()
    for i in range(frame_n):
        k_cur = []
        line1 = f.readline().split()
        line2 = f.readline().split()
        line3 = f.readline().split()
        k_cur.append(np.array([float(x) for x in line1], dtype=np.float32))
        k_cur.append(np.array([float(x) for x in line2], dtype=np.float32))
        k_cur.append(np.array([float(x) for x in line3], dtype=np.float32))

        r_cur = []
        line1 = f.readline().split()
        line2 = f.readline().split()
        line3 = f.readline().split()
        r_cur.append(np.array([float(x) for x in line1], dtype=np.float32))
        r_cur.append(np.array([float(x) for x in line2], dtype=np.float32))
        r_cur.append(np.array([float(x) for x in line3], dtype=np.float32))

        line1 = f.readline().split()
        t_cur = np.array([[float(x) for x in line1]], dtype=np.float32)
        f.readline()
        f.readline()
        K.append(k_cur)
        R.append(r_cur)
        T.append(t_cur)
    K = np.asarray(K)
    R = np.asarray(R)
    T = np.asarray(T)
    return K,R,T



def E_computation(video,K,R,T,drange,sigma_c,sigma_d,Frame_num=3):
    start = 3
    img_origin = cv2.imread("Road/src/"+video[start])
    w,h,c = img_origin.shape
    example = np.zeros([w, h])
    unary = np.tile(example[:, :, np.newaxis], [1, 1, len(drange)])

    for t_p in tqdm(range(start-Frame_num, start+Frame_num)):
        if t_p == start:
            continue
        img_t = cv2.imread("Road/src/"+video[t_p])
        part1 = get_part1(K[start],R[start],K[t_p],R[t_p])
        part2 = get_part2(K[t_p],R[t_p],T[start],T[t_p])
        for i in range(w):
            for j in range(h):
                for d in range(len(drange)):
                    deep = drange[d]
                    coo_old = np.array([[j, i, 1]])
                    coo_new = np.dot(part1, coo_old.T) + deep * part2
                    jnew = int(coo_new[0] / coo_new[2])
                    inew = int(coo_new[1] / coo_new[2])
                    norm2 = l2(np.array([j,i]),np.array([jnew,inew]))
                    pv = np.exp(norm2 / (-2 * sigma_d * sigma_d))
                    if inew >= 0 and inew < w and jnew >= 0 and jnew < h:
                        color_dist = distance(img_origin[i][j], img_t[inew][jnew])
                        pc = sigma_c/(sigma_c + color_dist)
                        unary[i][j][d] += pc*pv

    u = 1/(unary.max(axis=2,keepdims=True))
    res = 1-unary*u
    return res

K,R,T = get_KRT("Road/cameras.txt")
video = get_video("Road\src")

img = cv2.imread("Road/src/"+video[1])
print(img.shape)
drange = np.arange(-0.01,0.01,0.01/5)
unary = E_computation(video,K,R,T,drange,0.5,0.5)
print(unary.shape)

lamda = 0.01
smooth = lamda - lamda * np.eye(len(drange))
for i in range(len(drange)):
    for j in range(len(drange)):
        if i != j:
            smooth[i][j] = abs(i-j)*lamda

w,h,c = img.shape
labels = gco.cut_grid_graph_simple(unary, smooth, connect=8,n_iter=-1)
labels = np.reshape(labels, [w, h])
print("shape",labels.shape)
labels = labels
print(labels)
plt.figure(num=1, dpi=80, figsize=(8,6))
plt.imshow(labels,cmap="gray")
plt.axis("off")
plt.title("Best Result")
plt.show()



# d_num =np.arange(0,130,1)
# w,h,c = img_left.shape
# image = np.zeros([w,h])
# unary = np.tile(image[:, :, np.newaxis], [1, 1, len(d_num)])
#
# lamda = 0.025
# smooth = lamda - lamda * np.eye(len(d_num))




# labels = gco.cut_grid_graph_simple(unary, smooth, connect=8,n_iter=-1)
# labels = np.reshape(labels, [w, h])
# print("shape",labels.shape)
# labels = labels
# print(labels)
# plt.figure(num=1, dpi=80, figsize=(8,6))
# plt.imshow(labels,cmap="gray")
# plt.axis("off")
# plt.title("Best Result")
# plt.show()