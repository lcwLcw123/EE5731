# -*- code = utf-8 -*-
# @Time: 2022/10/23 16:11
# @Author: Chen Zigeng
# @File:part4liu.py
# @Software:PyCharm


import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import gco
import os


def homogeneous_coord_grid(h, w):
    X, Y = np.float32(np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        indexing='xy'))

    X, Y, Z = (np.expand_dims(X, axis=0),
               np.expand_dims(Y, axis=0),
               np.ones([1, h, w], np.float32))

    return np.concatenate([X, Y, Z], axis=0)


def conujugate_coordinates(K, R1, T1, R2, T2, coorsxy, d, h, w):

    coorsxy = np.reshape(coorsxy, [3, -1])
    d = np.reshape(d, [1, -1])

    depth = (T1 - T2).T * d
    remap = (K * R2.T) * ((R1 * K.I) * coorsxy + depth)
    remap = np.divide(remap, remap[2, :])
    remap = np.reshape(np.asarray(remap), [3, h, w])
    return remap


def L2_norm(img_a, img_b, keepdims=True):
    assert isinstance(img_a, np.ndarray)
    assert isinstance(img_b, np.ndarray)

    assert img_a.dtype == np.float32
    assert img_b.dtype == np.float32

    assert len(img_a.shape) == 3
    assert len(img_b.shape) == 3
    assert img_b.shape == img_a.shape

    return np.sqrt(np.sum(np.square(img_a - img_b), axis=-1, keepdims=keepdims))


def read_directory(directory_name):
    for filename in os.listdir(r"./" + directory_name):
        img = cv.imread(directory_name + "/" + filename)
        array_of_img.append(np.float32(img))


def load_depthmaps(directory, sequence):
    D = []
    for i in range(len(sequence)):
        depthmap_name = os.path.join(directory, "depth_" + str(i).zfill(4) + ".npy")

        depthmap = np.load(depthmap_name)
        D.append(depthmap)
    return D


def load_camera_params(filename):
    with open(filename, 'r') as f:
        frame_num = int(f.readline())
        f.readline()

        def read_vector():
            vec = f.readline().split()
            return np.array([float(x) for x in vec], dtype=np.float32)

        def read_matrix():
            r1 = read_vector()
            r2 = read_vector()
            r3 = read_vector()
            return np.matrix([r1, r2, r3], dtype=np.float32)

        K_sequence = [None] * frame_num
        R_sequence = [None] * frame_num
        T_sequence = [None] * frame_num

        for i in range(frame_num):
            K_sequence[i] = read_matrix()
            R_sequence[i] = read_matrix()
            T_sequence[i] = np.asmatrix(read_vector())
            f.readline()
            f.readline()
    return K_sequence, R_sequence, T_sequence


def get_smooth_simple(d_number):
    smooth = np.zeros((d_number, d_number))
    for i in range(d_number):
        for j in range(d_number):
            if i != j:
                smooth[i][j] = abs(i - j)
    return smooth


def get_depth_map(unary, smooth, lambda_factor=0.1):
    labels = gco.cut_grid_graph_simple(unary, smooth * lambda_factor, connect=4, n_iter=-1)
    print('labels')
    print(labels.shape)
    labels = labels.reshape(H, W)
    print(labels.shape)
    return labels


# Disparity Initialization
files = 'viedeo\Road\Road\src'
txt_path ='viedeo\Road\Road\cameras.txt'
out_dir = 'viedeo\depth'

K, R, T = load_camera_params(txt_path)
array_of_img = []
read_directory(files)

H, W, D = array_of_img[0].shape
disparity = np.arange(0, 0.02, 0.02 / 65)
d_number = len(disparity)

choose_frame = 1

sequence = np.array([5])
num_frame = sequence.shape[0]
L = np.zeros((H, W, d_number))

print(f'Disparity Initialization choose_fram =={choose_frame} sequence ={sequence}')
use_bundle = False
depth_map_save = []
img1 = array_of_img[choose_frame]

h, w = H, W
levels = d_number
depth_values = disparity
I_t = img1
sigma_c = 1.0
sigma_d = 2.5
x_h = homogeneous_coord_grid(h, w)  # [3, h, w]
d = np.zeros([h, w], dtype=np.float32)  # [h, w]
L = np.zeros([levels, h, w], dtype=np.float32)  # [levels, h, w]
for i in range(num_frame):
    print(f'Disparity Initialization for image {sequence[i]}')
    I_t_prime = array_of_img[sequence[i]]
    fram = sequence[i]
    for level in range(levels):

        d[:, :] = depth_values[level]

        x_prime_h = conujugate_coordinates(K[choose_frame], R[choose_frame], T[choose_frame], R[fram], T[fram], x_h, d,
                                           h, w)

        x_prime = np.transpose(x_prime_h[:2, :, :], [1, 2, 0])
        I_t_prime_projected = cv.remap(
            src=I_t_prime,
            map1=x_prime,
            map2=None,
            interpolation=cv.INTER_NEAREST,
            borderValue=[128, 128, 128])

        color_difference = L2_norm(I_t, I_t_prime_projected, keepdims=False)
        pc = sigma_c / (sigma_c + color_difference)

        if not use_bundle:
            L[level, :, :] += pc

    u = np.reciprocal(L.max(axis=0, keepdims=True))
    unary = 1 - u * L
    final_unary = np.zeros((H, W, d_number))
    for k in range(d_number):
        final_unary[:, :, k] = unary[k, :, :]

    smooth = get_smooth_simple(d_number)

    depthmap = get_depth_map(final_unary, smooth, lambda_factor=0.1)

    depthmap_filename = os.path.join(out_dir, "depth_" + str(i).zfill(4))

    np.save(depthmap_filename, depthmap)

    Max = np.float32(depthmap.max())
    cv.imwrite(depthmap_filename + '.png', np.uint8(depthmap / Max * 255))

# Using Boundle Optimization
use_bundle = True

print(f'Using Boundle Optimization choose_fram =={choose_frame} sequence ={sequence}')
for i in range(num_frame):
    I_t_prime = array_of_img[sequence[i]]
    print(f"solving img{sequence[i]}")
    for level in range(levels):

        d[:, :] = depth_values[level]

        x_prime_h = conujugate_coordinates(K[choose_frame], R[choose_frame], T[choose_frame], R[fram], T[fram], x_h, d,
                                           h, w)

        x_prime = np.transpose(x_prime_h[:2, :, :], [1, 2, 0])
        I_t_prime_projected = cv.remap(
            src=I_t_prime,
            map1=x_prime,
            map2=None,
            interpolation=cv.INTER_NEAREST,
            borderValue=[128, 128, 128])

        color_difference = L2_norm(I_t, I_t_prime_projected, keepdims=False)
        pc = sigma_c / (sigma_c + color_difference)

        if not use_bundle:

            L[level, :, :] += pc

        else:

            D = load_depthmaps(out_dir, sequence)
            depth_indices = D[i]  # get prev. estimated depth
            depth_indices_projected = cv.remap(
                src=depth_indices,
                map1=x_prime,
                map2=None,
                interpolation=cv.INTER_NEAREST,
                borderValue=int(levels / 2.0))
            print(depth_values)
            print(depth_indices_projected)
            np.take(depth_values, depth_indices_projected, out=d)

            projected_x_prime_h = conujugate_coordinates(K[fram], R[fram], T[fram], R[choose_frame], T[choose_frame],
                                                         x_prime_h, d, h, w)

            color_difference_norm = np.sum(
                np.square(x_h - projected_x_prime_h),
                axis=0,
                keepdims=False)

            pv = np.exp(color_difference_norm / (-2 * sigma_d * sigma_d))

            L[level, :, :] += pc * pv

u = np.reciprocal(L.max(axis=0, keepdims=True))
unary = 1 - u * L
final_unary = np.zeros((H, W, d_number))
for i in range(d_number):
    final_unary[:, :, i] = unary[i, :, :]

smooth = get_smooth_simple(d_number)

print('get_depth_map................................')
depthmap = get_depth_map(final_unary, smooth, lambda_factor=0.1)
plt.figure(num=1, dpi=100, figsize=(6, 6))
plt.imshow(depthmap, cmap='gray')
plt.title('Best Result for Depth from Video by using boundle optimization')
plt.axis("off")
plt.show()