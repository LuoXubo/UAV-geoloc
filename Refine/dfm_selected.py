#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:09:25 2021

@author: ufukefe
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


dataset_root = './drone-satellite/result_0.5/80'

folder_list = os.listdir(dataset_root)
folder_list.sort()
folder_list.sort(key=lambda x:int(x[:-1]))
#print(folder_list)
index = 0
for folder in folder_list:
    nextroot = os.path.join(dataset_root, folder)
    files = os.listdir(nextroot)
    #print(files)
    im1 = Image.open(nextroot + '/' + files[0])
    im2 = Image.open(nextroot + '/' + files[1])
    # scale images smaller for matching (GPU may out of memory if given high resolutions)
    im1_new = im1.resize((1000,1000))
    im2_new = im2.resize((1000,1000))
    #print(im2_new.size)
    img_A = np.array(im1_new)
    img_B = np.array(im2_new)

    start = time.time()
    H, H_init, points_A, points_B = fm.match(img_A, img_B)
    end = time.time()
    #print(H)
    

    total_time = total_time + (end - start)
    total_pairs = total_pairs + 1
    
    keypoints0 = points_A.T
    keypoints1 = points_B.T
    
    mtchs = np.vstack([np.arange(0,keypoints0.shape[0])]*2).T
    
    if files[0].count('/') > 0:
    
        p1 = files[0].split('/')[files[0].count('/')].split('.')[0]
        p2 = files[1].split('/')[files[0].count('/')].split('.')[0]
        
    elif files[0].count('/') == 0:
        p1 = files[0].split('.')[0]
        p2 = files[1].split('.')[0]

    if(len(mtchs) >= 10):    
        np.savez_compressed(config['output_directory'] + '/' + p1 + '_' + p2 + '_' + 'matches' + '_' + str(index), 
                        keypoints0=keypoints0, keypoints1=keypoints1, matches=mtchs)
    
        if config['display_results']: 
            cv2.imwrite(config['output_directory'] + '/' + p1 + '_' + p2 + '_' + 'matches' + '_' + str(index) + '.png',
                    draw_matches(img_A, img_B, keypoints0, keypoints1))

        np.save(config['output_directory'] + '/' + p1 + '_' + p2 + '_' + 'Homo' + '_' + str(index) + '.npy', H)
        
    index += 1

        
print(f'n \n \nAverage time is: {round(1000*total_time/total_pairs,0)} ms' )    
print(f'Results are ready in ./{config["output_directory"]} directory\n \n \n' )

