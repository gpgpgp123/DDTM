import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import rasterio
import os
import torch

LABEL_CLASS_COLORMAP = { # Color map for Chesapeake dataset
    0: (0, 0, 0),
    1: (222, 34, 7),
    2: (34, 97, 38),
    3: (0, 255, 36),
    4: (70, 107, 159)
}

RGB_TO_LABEL = {v: k for k, v in LABEL_CLASS_COLORMAP.items()}

def fast_hist(a, b, n): 
    k = (a > 0) & (a < n)
    matrx=np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2)

    if len(matrx)>25:
        return matrx[len(matrx)-26:-1].reshape(n,n)
    else :
        return matrx.reshape(n,n)


def per_class_iu(hist): 
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  

def per_class_Freq(hist):  
    return np.sum(hist, axis=1) / np.sum(hist)

def per_class_OA(hist):
    return np.diag(hist) / np.sum(hist,axis=1)

def per_class_kappa(hist):
    hist_sum=np.sum(hist)
    p0=np.sum(np.diag(hist))/np.sum(hist)
    pe=np.sum(np.sum(hist,axis=0)*np.sum(hist,axis=1))/(hist_sum*hist_sum)
    return (p0-pe)/(1-pe)

def label_mapping(input, mapping):  
    output = np.copy(input)  
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]  
    return np.array(output, dtype=np.int64)  



def compute_mIoU(gt_dir, pred_dir, devkit_dir, val_txt, label_txt):  
    # print(os.getcwd())
    os.chdir("/mnt/disk1/gp/Paraformer/configs")
    """
    Compute IoU given the predicted colorized images and
    """
    with open(join(devkit_dir, 'NLCD.json'), 'r') as fp:  
        info = json.load(fp)
    num_classes = np.int32(info['classes'])
    print('Num classes', num_classes)  
    name_classes = np.array(info['label'], dtype=str)
    hist = np.zeros((num_classes, num_classes)) 

    image_path_list = join(devkit_dir, val_txt)  
    label_path_list = join(devkit_dir, label_txt)  
    gt_imgs = open(label_path_list, 'r').read().splitlines()  
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]  
    pred_imgs = open(image_path_list, 'r').read().splitlines()  
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]  
    # pred_imgs=pred_imgs[0:-2]
    for ind in range(len(gt_imgs)):
        with rasterio.open(pred_imgs[ind]) as f:
            r = f.read(1)  # 红通道
            g = f.read(2)  # 绿通道
            b = f.read(3)  # 蓝通道
            # 堆叠成 (H, W, 3) 的数组
            rgb_image = np.stack([r, g, b], axis=-1)
            height, width, _ = rgb_image.shape
            labels = np.zeros((height, width), dtype=np.uint8)
            # 将图像中的每个 RGB 值映射到标签
            for label, rgb in LABEL_CLASS_COLORMAP.items():
                mask = np.all(rgb_image == rgb, axis=-1)
                labels[mask] = label
            print(pred_imgs[ind])
        pred = np.array(labels) 
        with rasterio.open(gt_imgs[ind]) as f1:
            label = f1.read(1)
        label = np.array(label)  
        if len(label.flatten()) != len(pred.flatten()):  
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  
        if ind > 0 and ind % 10 == 0:  
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.nanmean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    Freq = per_class_Freq(hist)
    Kappa = per_class_kappa(hist)
    Oa = per_class_OA(hist)
    for ind_class in range(num_classes):  
        print('>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2))+':\t'+str(round(Freq[ind_class]* 100, 2)))
    print('> mIoU: '+str((1/4 * mIoUs[Freq > 0]).sum()))
    print('> FWIoU: '+str((Freq[Freq > 0] * mIoUs[Freq > 0]).sum()))
    print('===> OA: '+str((1/4 * Oa[Freq > 0]).sum()))
    print('===> Kappa: '+str(Kappa))
    return mIoUs

Poland_label='/home/ashelee/Dataset/label_color_uni-4class'
Poland_Conf='/home/ashelee/Dataset/Polandconfig-4class'

VA_CCLC='/home/ashelee/NAIP_DATA/CCLC-VA-4class'
VA_Conf='/home/ashelee/NAIP_DATA/VAconfig'

NY_CCLC='/home/ashelee/NAIP_DATA/CCLC-NY-4class'
NY_Conf='/home/ashelee/NAIP_DATA/NYconfig'

PA_CCLC='/home/ashelee/NAIP_DATA/CCLC-PA-4class'
PA_Conf='/home/ashelee/NAIP_DATA/PAconfig'

DE_CCLC='/home/ashelee/NAIP_DATA/CCLC-DE-4class'
DE_Conf='/home/ashelee/NAIP_DATA/DEconfig'

ML_CCLC='/home/ashelee/NAIP_DATA/CCLC-ML-4class'
ML_Conf='/home/ashelee/NAIP_DATA/MLconfig'

WV_CCLC='/home/ashelee/NAIP_DATA/CCLC-WV-4class'
WV_Conf='/home/ashelee/NAIP_DATA/WVconfig'
def main():
    compute_mIoU('../dataset/CP/wv/test/HR_unify_label',
    '../results/CP/wv/2025-07-23_23-41-21/unify_map',
     './config',
    'wv/val.txt',
     'wv/label.txt')
main()