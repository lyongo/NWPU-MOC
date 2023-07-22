# -*- coding: utf-8 -*-

import os
from importlib import import_module
import misc.transforms as own_transforms
import torchvision.transforms as standard_transforms
from . import basedataset
from . import setting
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from config import cfg
from misc.utils import train_collate


def createTrainData(datasetname, Dataset, cfg_data):

    folder, list_file = None, None

    if datasetname in ['MOC_RS','IC15']:
        list_file=[]
        list_file.append({'data_path':cfg_data.DATA_PATH,
                          'imgId_txt': cfg_data.TRAIN_LST})
    
    else:
        print('dataset is not exist')

    main_transform = own_transforms.Compose([
        own_transforms.ScaleByRateWithMin([0.8, 1.2], cfg_data.TRAIN_SIZE[0], cfg_data.TRAIN_SIZE[1]),
        own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
        own_transforms.RandomHorizontallyFlip(),
    ])

    rgb_transform = standard_transforms.Compose([
        own_transforms.RGB2Gray(0.1, 3),
        own_transforms.GammaCorrection([0.4,2]),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD_RGB)
    ])

    ir_transform = standard_transforms.Compose([
        own_transforms.RGB2Gray(0.1, 3),
        own_transforms.GammaCorrection([0.4,2]),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD_IR)
    ])

    gt_transform = standard_transforms.Compose([
        own_transforms.GTScaleDown(cfg_data.LABEL_FACTOR),
        standard_transforms.ToTensor(),
  
        own_transforms.LabelNormalize(cfg_data.LOG_PARA)
    ])
    

    train_set = Dataset(datasetname, 'train',
        main_transform = main_transform,
        rgb_transform = rgb_transform,
        ir_transform = ir_transform,
        gt_transform = gt_transform,
        # mask_transform = mask_transform,
        list_file = list_file
    )
    if datasetname in ['MOC_RS', 'IC15']:
        return DataLoader(train_set, collate_fn=train_collate, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=6, shuffle=True, drop_last=True)
    else:
        return 'error'

def createValData(datasetname, Dataset, cfg_data):

    if datasetname in ['MOC_RS']:
        list_file=[]
        list_file.append({'data_path':cfg_data.DATA_PATH,
                          'imgId_txt': cfg_data.VAL_LST})
    elif datasetname in ['IC15']:
        list_file=[]
        list_file.append({'data_path':cfg_data.VAL_PATH,
                          'imgId_txt': cfg_data.VAL_LST})
    else:
        print('dataset is not exist')


    rgb_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD_RGB)
    ])

    ir_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD_IR)
    ])

    gt_transform = standard_transforms.Compose([
        own_transforms.GTScaleDown(cfg_data.LABEL_FACTOR),
        standard_transforms.ToTensor(),
        own_transforms.LabelNormalize(cfg_data.LOG_PARA),      
    ])
    

    val_set = Dataset(datasetname, 'val',
        rgb_transform = rgb_transform,
        ir_transform = ir_transform,
        gt_transform = gt_transform,
        # mask_transform = mask_transform,
        list_file = list_file
    )

    if datasetname in ['MOC_RS']:
        return DataLoader(val_set, collate_fn=train_collate, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=6, shuffle=True, drop_last=True)
    else:
        return 'error'

def createTestData(datasetname, Dataset, cfg_data):

    if datasetname in ['MOC_RS']:
        list_file=[]
        list_file.append({'data_path':cfg_data.DATA_PATH,
                          'imgId_txt': cfg_data.TEST_LST})  
    else:
        print('dataset is not exist')


    rgb_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD_RGB)
    ])

    ir_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD_IR)
    ])

    gt_transform = standard_transforms.Compose([
        own_transforms.GTScaleDown(cfg_data.LABEL_FACTOR), 
        standard_transforms.ToTensor(),
        own_transforms.LabelNormalize(cfg_data.LOG_PARA),      
    ])
    


    test_set = Dataset(datasetname, 'test',
        rgb_transform = rgb_transform,
        ir_transform = ir_transform,
        gt_transform = gt_transform,
        # mask_transform = mask_transform,
        list_file = list_file
    )

    if datasetname in ['MOC_RS']:
        return DataLoader(test_set, collate_fn=train_collate, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=6, shuffle=True, drop_last=True)
    else:
        return 'error'


def createRestore(mean_std):
    return standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

def loading_data(datasetname):
    datasetname = datasetname.upper() 
    cfg_data = getattr(setting, datasetname).cfg_data
    Dataset = basedataset.Dataset        
    train_loader = createTrainData(datasetname, Dataset, cfg_data)
    val_loader = createValData(datasetname, Dataset, cfg_data)
    return train_loader, val_loader

def loading_test_data(datasetname):
    datasetname = datasetname.upper() 
    cfg_data = getattr(setting, datasetname).cfg_data
    Dataset = basedataset.Dataset        
    test_loader = createTestData(datasetname, Dataset, cfg_data)
    return test_loader