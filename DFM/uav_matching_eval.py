import cv2
import os
import numpy as np
import time
import torchvision.transforms as transforms
from numpy import average, dot, linalg
from PIL import Image
from sklearn import metrics as mr
import  skimage.metrics as metrics
import yaml
from tqdm import tqdm
from DeepFeatureMatcher import DeepFeatureMatcher
import math

central_coords_x = 500
central_coords_y = 500
pt_drone = np.matrix([int(central_coords_x/2), int(central_coords_y/2), 1])

root_dataset = '/home/zino/lxb/LPNDFM/datasets/dataset8/'
res_XY = root_dataset + 'uav/res_eval.txt'
gt_XY = root_dataset + 'uav/gt.txt'
save_path = root_dataset + 'dfm_matches/'
root_images = root_dataset + 'lpn_results/'


def transformation(pt_drone, H):
    return H @ pt_drone

def coords(mat):
    a = mat[0][0]/mat[2][0]
    b = mat[1][0]/mat[2][0]
    return float(a), float(b)

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

def mutual_similarity(imgA, imgB):
    # return mr.normalized_mutual_info_score(imgA, imgB)
    return mr.normalized_mutual_info_score(imgA.reshape(-1), imgB.reshape(-1))

def ncc_similarity(imgA, imgB):
    return np.mean(np.multiply((imgA-np.mean(imgA)), (imgB-np.mean(imgB))))/(np.std(imgA)*np.std(imgB))

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def cos_similarity(imgA, imgB):
    images = [imgA, imgB]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image:
            # for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res

def ssim_similarity(imgA, imgB):
    return metrics.structural_similarity(imgA, imgB, channel_axis = 1)
        
def calc(sate_img, uav_img, matches):
    if (len(sate_img.shape) == 3):
        sate_img = cv2.cvtColor(sate_img, cv2.COLOR_BGR2GRAY)
    if (len(uav_img.shape) == 3):
        uav_img = cv2.cvtColor(uav_img, cv2.COLOR_BGR2GRAY)

    mutual_score = mutual_similarity(sate_img, uav_img)
    ssim_score = ssim_similarity(sate_img, uav_img)

    return sigmoid(matches/100-2) *sigmoid( ssim_score) * sigmoid (mutual_score * 10)


with open("config.yml", "r") as configfile:
    config = yaml.safe_load(configfile)['configuration']

# Make result directory
# os.makedirs(config['output_directory'], exist_ok=True)

# Construct FM object
fm = DeepFeatureMatcher(enable_two_stage=config['enable_two_stage'], model=config['model'],
                        ratio_th=config['ratio_th'], bidirectional=config['bidirectional'], )

with open(gt_XY, 'r') as f:
    gtlines = f.readlines()
f.close()

uav_folder_list = os.listdir(root_images)
uav_folder_list.sort()

cnt = 0
for uav_folder in tqdm(uav_folder_list):
    imgs = os.listdir(root_images + uav_folder)
    imgs.sort()
    best_score = 0
    best_mtch = 0
    im1 = cv2.imread(root_images + '/' + uav_folder + '/' + imgs[10])
    img_A = np.array(im1)
    img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)

    subimg_candidate = imgs[0:10]
    # 查找最匹配的遥感子图
    start = time.time()
    print('Begin to search best sub-satellite imge for uav image: %s ......'%uav_folder)

    # 在遥感子图中找
    for satellite_folder in subimg_candidate: # satellite_folder: Rank01-45_1000_1000.png
        im2 = cv2.imread(root_images + '/'  + uav_folder + '/' + satellite_folder)
        img_B = np.array(im2)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
    
        H, H_init, points_B, points_A = fm.match(img_B, img_A)
        
        keypoints0 = points_A.T
        keypoints1 = points_B.T
        mtchs = np.vstack([np.arange(0, keypoints0.shape[0])]*2).T  

        # 将uav图像对齐到satellite图像
        warped_uav_img = cv2.warpPerspective(img_A, H, (img_B.shape[0], img_B.shape[1]))

        # 通过阈值分割，将转换后的uav轮廓分离出来
        gray_warped_uav = cv2.cvtColor(warped_uav_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_warped_uav, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            polygon = cv2.approxPolyDP(contour, 10, True)
            polygons.append(polygon)

        mask1  = np.zeros([img_B.shape[0], img_B.shape[1]], np.uint8)
        mask2  = np.zeros([img_B.shape[0], img_B.shape[1]], np.uint8)

        cv2.fillPoly(mask1, polygons, 255)
        cv2.fillPoly(mask2, polygons, 255)

        # 将提取出的uav轮廓应用到uav和satellite图像上，分离出公共部分来
        masked_image1 = cv2.bitwise_and(img_B, img_B, mask=mask1)
        masked_image2 = cv2.bitwise_and(warped_uav_img, warped_uav_img, mask=mask2)


        matching_score = calc(masked_image2,masked_image1, len(mtchs))
        if matching_score > best_score:
            best_kpA = keypoints0
            best_kpB = keypoints1
            best_mtch = len(mtchs)
            best_score = matching_score
            best_sub = satellite_folder
            best_H = H


    startX, startY = best_sub.split('.')[0].split('_')[1:3]
    pt_sate = transformation(pt_drone.T, best_H)

    x, y = coords(pt_sate)


    resX, resY = float(startX) + coords(pt_sate)[0], float(startY) + coords(pt_sate)[1]

    gtline = gtlines[cnt].split('\n')[0]
    timestamp = gtline.split(' ')[0] # 直接使用gt.txt里已经除以500后的时间戳
    z, qw, qx, qy, qz = gtline.split(' ')[3:]
    cnt = cnt + 1

    im2 = cv2.imread(root_images + '/'  + uav_folder + '/' + best_sub)
    img_B = np.array(im2)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
    
    warped_uav_img = cv2.warpPerspective(img_A, best_H, (img_B.shape[0], img_B.shape[1]))
    added_img = cv2.addWeighted(warped_uav_img, 0.7, img_B, 0.3, 0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cv2.imwrite(save_path + uav_folder + '.png', added_img)

    cv2.imwrite(save_path + uav_folder + '_matches.png', draw_matches(img_A, img_B, best_kpA, best_kpB))

    with open(res_XY, 'a+') as f:
        # f.write('%s %f %f %s %s %s %s %s\n'%(timestamp, resX, resY, z, qw, qx, qy, qz))
        f.write('%s %f %f %s %s %s %s %s %f\n'%(timestamp, resX, resY, z, qw, qx, qy, qz, best_score))
    end = time.time()
    print('Timestamp: %s        Position:{%f, %f}       matching points: %d     matching score:%f'%(timestamp, resX, resY, best_mtch, best_score))
