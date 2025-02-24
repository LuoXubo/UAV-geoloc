import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
# from skimage.transform import resize
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
# parser.add_argument('--query_index', default=0, type=int, help='test_image_index')
parser.add_argument('--test_dir', default='./data/test/',
                    type=str, help='./test_data')
parser.add_argument('--Mars', default='Mars4', type=str, help='Scene name')
opts = parser.parse_args()


gallery_name = 'gallery_'+opts.Mars+'_satellite'
query_name = 'query_'+opts.Mars+'_drone'
# gallery_name = 'gallery_drone'
# query_name = 'query_satellite'

data_dir = opts.test_dir
# print(os.listdir(os.path.join(data_dir, query_name)))
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in [
    gallery_name, query_name]}

#####################################################################
# Show query


def imshow_scale(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    im = resize(im, (im.shape[0], im.shape[1] * 2), anti_aliasing=True)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated

# Show result


def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated


######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images


def sort_img(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index = np.argwhere(gl == -1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


for i in range(len(query_feature)):  
    print(i)
    # 类别写死了，为1，下面的参数
    index = sort_img(query_feature[i], 1, gallery_feature, gallery_label)

    ########################################################################
    # Visualize the rank result
    print(image_datasets[query_name].imgs[i])
    query_path, _ = image_datasets[query_name].imgs[i]
    # 类别写死了，为1
    query_label = 1
    print(query_path)
    # done->satellite: retrive the filename of satellite image blocks
    block = query_path.split('/')[-1].split('.')[0]
    # print('Top 10 images are as follow:')
    # define filename as same as the croped satellite blocks
    save_folder = '/home/zino/lxb/Image-Matching-codes/datasets/'+opts.Mars+'/lpn_results/%s' % block

    # # define index started from 0 as filename

    if not os.path.isdir(save_folder):
        # os.mkdir(save_folder)
        os.makedirs(save_folder)
    os.system('cp %s %s/query.png' % (query_path, save_folder))


    try:  # Visualize Ranking Result
        for i in range(10):
            # ax = plt.subplot(1, 11, i+2)
            # ax.axis('off')
            img_path, _ = image_datasets[gallery_name].imgs[index[i]]
            label = gallery_label[index[i]]
            # print(label)
            # imshow(img_path)
            num = img_path.split('/')[-1].split('.')[0]
            # os.system('cp %s %s/%02d.jpg'%(img_path, save_folder, int(num)))
            os.system('cp %s %s/Rank%02d-%s.png' %
                      (img_path, save_folder, i+1, num))
            # if label == query_label:
            #     ax.set_title('%d' % (i+1), color='green')
            # else:
            #     ax.set_title('%d' % (i+1), color='red')
        # plt.pause(100)  # pause a bit so that plots are updated
    except RuntimeError:
        for i in range(10):
            img_path = image_datasets.imgs[index[i]]
            # print(img_path[0])
        # print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
    # fig.savefig(save_folder+"/show.png", dpi=192)
