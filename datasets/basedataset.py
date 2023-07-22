# -*- coding: utf-8 -*-

from unicodedata import category
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import torch
import math
import json
import cv2

from misc import transforms
from torchvision.transforms import transforms as standard_transforms

class Dataset(data.Dataset):
    def __init__(self, datasetname, mode, **argv):
        self.mode = mode
        self.datasetname = datasetname

        self.img_paths = []
        self.gt_paths = [] 
        self.info = []

        for data_infor in argv['list_file']:
            data_path, imgId_txt = data_infor['data_path'], data_infor['imgId_txt']
            with open(os.path.join(data_path, imgId_txt)) as f:
                img_names = f.readlines()
            if "MOC" in data_path:

                for img_name in img_names:
                    img_name=img_name.strip() 
                    rgb_path = os.path.join(data_path, 'rgb', img_name)
                    ir_path = os.path.join(data_path, 'ir', img_name.replace("orth25", "ir")) 
                    self.img_paths.append([rgb_path, ir_path])
                    self.gt_paths.append(os.path.join(data_path, 'gt', img_name.replace("png", "npz")))

            elif "ASPDNet_dataset" in data_path: #IC15
                 for line in img_names:
                    line=line.strip()
                    self.img_paths.append(os.path.join(data_path,  'images',line))
                    self.gt_paths.append(os.path.join(data_path,  'dots', line[:-4]+ '.png'))
        
        self.num_samples = len(self.img_paths)
        self.main_transform = None
        if 'main_transform' in argv.keys():
            self.main_transform = argv['main_transform']
        self.rgb_transform = None
        if 'rgb_transform' in argv.keys():
            self.rgb_transform = argv['rgb_transform']
        self.ir_transform = None
        if 'ir_transform' in argv.keys():
            self.ir_transform = argv['ir_transform']
        self.gt_transform = None
        if 'gt_transform' in argv.keys():
            self.gt_transform = argv['gt_transform']
        self.mask_transform = None
        if 'mask_transform' in argv.keys():
            self.mask_transform = argv['mask_transform']

        if self.mode == 'train':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} train images.')
        if self.mode == 'val':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} validation images.')
        if self.mode == 'test':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} test images.')

    
    def __getitem__(self, index):

        rgb, nir, gt_map = self.read_data(index)

        if self.main_transform != None:
            rgb, nir, gt_map  = self.main_transform(rgb, nir, gt_map)
        
        if self.rgb_transform != None:
            rgb = self.rgb_transform(rgb)
        if self.ir_transform != None:
            nir = self.ir_transform(nir)
        if self.gt_transform != None:
            gt_map = self.gt_transform(gt_map)

        if self.mode == 'train' or self.mode == 'val' or self.mode == 'test':    

            return rgb, nir, gt_map
        else:
            print('invalid data mode!!!')

    def __len__(self):
        return self.num_samples

    def read_data(self,index):

        rgb_path, ir_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        rgb = Image.open(rgb_path)
        ir = Image.open(ir_path)
        if rgb.mode != 'RGB':
            rgb=rgb.convert('RGB')
        if ir.mode != 'RGB':
            ir=ir.convert('RGB')

        gt_map = np.load(gt_path)['arr_0'] # [H, W, classes]
        
        if gt_map.shape[0] != 1024 or gt_map.shape[1] !=1024:
            raise ValueError("gt_map  shape error:", gt_path)
  
        return rgb, ir, gt_map

    def get_num_samples(self):
        return self.num_samples








