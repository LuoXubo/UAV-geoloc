#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 10:04:25 2021

@author: wwp
@revised:tyl

Based on the homography estimation, this file aims to find the 
position of the central coordinates of the drone images in the
 satellite images
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import *

dataset_root = "./bestMatches4eval/DFM_homo_npy"

folder_list = os.listdir(dataset_root)

images_root = "./bestMatches4eval/img_pairs"
image_folders = os.listdir(images_root)
image_folders.sort()

central_coords_x = 2172
central_coords_y = 1448

pt_drone = np.matrix([int(central_coords_x/2),int(central_coords_y/2),1])

def transformation(pt_drone, H):
    return H @ pt_drone

def coords(mat):
    a = mat[0][0]/mat[2][0]
    b = mat[1][0]/mat[2][0]
    return float(a),float(b)

res = {}
for folder in folder_list:
    # print(folder)
    if '.npy' in folder: 
        # print(folder)
        data = np.load(dataset_root + '/' + folder)
        # print(data) 
        H = np.matrix(data)
        # print(H)

        # output the series of central points 
        pt_sate = transformation(pt_drone.T, H) # PIL和matlab之间的H转置差异,(pt_drone @ H.T).T = H @ pt_drone.T 
        # print(pt_sate)
        x,y = coords(pt_sate)

        # find the position of the satellite images in the background
        # num = int(folder.split('_')[-1].split('.')[0])
        num = int(folder.split('_')[0])
        for folder2 in image_folders:
            # print(folder2)
            if num == int(folder2):
                # print(folder2)
                nextroot = os.path.join(images_root, folder2)
                files = os.listdir(nextroot)
                files.sort()
                # print(files[0])
                y0 = int(files[-1].split('_')[1])
                x0 = int(files[-1].split('_')[-1].split('.')[0])
                # print(x0,y0)
        
        zb = [x+x0, y+y0]
        res[num] = zb

res2 = dict(sorted(res.items(), key = lambda x:x[0]))

gtruth = {}
a = [] 
b = []
for line in open("./GT.txt"): 
    a.append(float(line.split(' ')[1]))
    b.append(float(line.split(' ')[-1]))
    label = int(line.split(' ')[0])
    gtzb = [float(line.split(' ')[1]), float(line.split(' ')[-1])]
    gtruth[label] = gtzb

gtruth2 = dict(sorted(gtruth.items(), key = lambda x:x[0]))

x = []
y = []
for item in res2.keys():
    x.append(res2[item][0])
    y.append(res2[item][1])
plt.plot(x,y,'r*',label = 'result')
plt.plot(a,b,label = 'groundtruth')
plt.legend()
plt.title("Trajectory comparison")
plt.savefig("LPN+DFM_comparison.png") 
plt.show()

# calculate the difference between the results and the groundtruth 
dist = []
err = {}

# print(res2)
# print(gtruth2)
for i in res2.keys():
    coord_res = res2[i]
    coord_gt = gtruth2[i]
    currentdist = math.sqrt((coord_res[0] - coord_gt[0])**2 + (coord_res[1] - coord_gt[1])**2)
    dist.append(currentdist)
    err[i] = [abs(coord_res[0] - coord_gt[0]), abs(coord_res[1] - coord_gt[1])]
    print("current img:" + str(i) + " current dist:" + str(currentdist))

print("mean dist:" + str(mean(dist)))
 
# draw the error curve of x and y 
err2 = sorted(err.items(), key = lambda x:x[0])

x_err = []
y_err = []
pos = []
for item in err2:
    pos.append(item[0])
    x_err.append(item[1][0])
    y_err.append(item[1][1])

pose = np.arange(0,11,1)
plt.plot(pose, x_err, label = 'x_error')
plt.plot(pose, y_err, label = 'y_error')
plt.ylabel('drone position error(pixel)') 
plt.xlabel('the xth drone image') 
plt.legend()
plt.savefig("LPN+DFM_error.png") 
plt.show()