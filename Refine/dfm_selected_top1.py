#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:09:25 2021

@author: ufukefe

@revised: tianyaolin

说明： 根据selected.py修改而来,从R@10中筛选出来最优匹配(根据特征点的多少来筛选的)
"""

import os
import argparse
import yaml
import cv2
from DeepFeatureMatcher import DeepFeatureMatcher
from PIL import Image
import numpy as np
import time

#To draw_matches
def draw_matches(img_A, img_B, keypoints0, keypoints1):
    
    p1s = []
    p2s = []
    dmatches = []
    for i, (x1, y1) in enumerate(keypoints0):
         
        p1s.append(cv2.KeyPoint(x1, y1, 1))
        p2s.append(cv2.KeyPoint(keypoints1[i][0], keypoints1[i][1], 1))
        j = len(p1s) - 1
        dmatches.append(cv2.DMatch(j, j, 1))
        
    matched_images = cv2.drawMatches(cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR), p1s, 
                                     cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR), p2s, dmatches, None)
    
    return matched_images

#Take arguments and configurations
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm wrapper for Image Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_pairs', type=str)

    args = parser.parse_args()  

with open("config.yml", "r") as configfile:
    config = yaml.safe_load(configfile)['configuration']
    
# Make result directory
os.makedirs(config['output_directory'], exist_ok=True)     
        
# Construct FM object
fm = DeepFeatureMatcher(enable_two_stage = config['enable_two_stage'], model = config['model'], 
                    ratio_th = config['ratio_th'], bidirectional = config['bidirectional'], )
    

total_time = 0
total_pairs = 0

dataset_root = '../../datasets/sanjiang/lpn_results/1/'

folder_list = os.listdir(dataset_root)
folder_list.sort()
print(folder_list)
folder_list.sort(key=lambda x:int(x[:-1]))
#print(folder_list)
index = 0
maxMatches = 0
descriptors = []
filename = []
matchImage = []
for folder in folder_list:
    nextroot = os.path.join(dataset_root, folder)
    files = os.listdir(nextroot)
    files.sort(key=lambda x: int(x[:2]))
    #print(files)
    source_file = files[0] # drone (wrapped image/query)
    target_file = files[1] # satellite
    # PIL和opencv的图像读取之间是转置关系，为了和matlab保持一致这里选择用opencv读取图片
    # 虽然可以用opencv进行图片读取，但色彩存在的问题影响了DFM算法的准确度
    im1 = Image.open(nextroot + '/' + source_file)
    w_im1, h_im1 = im1.size
    im2 = Image.open(nextroot + '/' + target_file)
    w_im2, h_im2 = im2.size
    # scale images smaller for matching (GPU may out of memory if given high resolutions)
    scale = 2 # 图像降采样尺度，降采样为原来的1/4，drone的尺度变为w_1086,h_724，satellite的尺度为w_1086,h_1086
    im1_new = im1.resize((int(w_im1/scale),int(h_im1/scale)))
    im2_new = im2.resize((int(w_im2/scale),int(h_im2/scale)))
    #print(im2_new.size)
    img_A = np.array(im2_new) # registered/fixed image
    img_B = np.array(im1_new) # wrapped image (* H)
    
    start = time.time()
    H, H_init, points_A, points_B = fm.match(img_A, img_B, display_results=0) # (dst, src), display_result用于查看wrap的映射关系
    end = time.time()
    # print("H':")
    # print(H)
    # Homography对角线上的元素为缩放参数（长宽比Aspect Ratio调整），h11为水平方向缩放参数，h22为垂直方向缩放参数
    # 还原缩放参数，与真值标注的H保持相同的映射
    diag=[1/scale,1/scale,1]
    scale_matrix = np.diag(diag)
    H = np.dot(np.dot(np.linalg.inv(scale_matrix),H), scale_matrix)
    # print("H:")
    # print(H)

    total_time = total_time + (end - start)
    total_pairs = total_pairs + 1
    
    keypoints0 = points_A.T
    keypoints1 = points_B.T
    
    mtchs = np.vstack([np.arange(0,keypoints0.shape[0])]*2).T
    
    if files[0].count('/') > 0:
    
        p1 = source_file.split('/')[files[0].count('/')].split('.')[0]
        p2 = target_file.split('/')[files[0].count('/')].split('.')[0]
        
    elif files[0].count('/') == 0:
        p1 = source_file.split('.')[0]
        p2 = target_file.split('.')[0]
    
    if(len(mtchs) > maxMatches): 
        maxMatches = len(mtchs)
        descriptors.append(keypoints1)
        descriptors.append(keypoints0)
        descriptors.append(mtchs)
        descriptors.append(H)
        filename.append(p1)
        filename.append(p2)
        matchImage.append(img_B)
        matchImage.append(img_A)
        
    index += 1

np.savez_compressed(config['output_directory'] + '/' + filename[-2] + '_' + filename[-1] + '_' + 'matches' + '_' + str(maxMatches), 
                keypoints0=descriptors[-4], keypoints1=descriptors[-3], matches=descriptors[-2])
    
if config['display_results']: 
    cv2.imwrite(config['output_directory'] + '/' + filename[-2] + '_' + filename[-1] + '_' + 'matches' + '_' + str(maxMatches) + '.png',
            draw_matches(matchImage[-2], matchImage[-1], descriptors[-4], descriptors[-3]))

np.save(config['output_directory'] + '/' + filename[-2] + '_' + filename[-1] + '_' + 'Homo' + '_' + str(maxMatches) + '.npy', descriptors[-1])

print(f'n \n \nAverage time is: {round(1000*total_time/total_pairs,0)} ms' )    
print(f'Results are ready in ./{config["output_directory"]} directory\n \n \n' )

