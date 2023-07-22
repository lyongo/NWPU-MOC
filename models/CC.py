import torch
import torch.nn as nn
import torch.nn.functional as F
from . import counters
from misc import layer
import numpy as np

from models.losses.cos_sim import CosSim_Loss

from config import cfg


class CrowdCounter(nn.Module):
    def __init__(self, net_name, gpu_id):
        super(CrowdCounter, self).__init__()
        

        self.net_name = net_name
        ccnet =  getattr(getattr(counters, net_name), net_name)
                
        gs_layer = getattr(layer, 'Gaussianlayer')
        
        self.MM = cfg.MM
        self.alpha = cfg.alpha
        self.CCN = ccnet()
        
        self.gs = gs_layer()
        
        if len(gpu_id) > 1:
            ids = range(len(gpu_id))  
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=ids).cuda()
            self.gs = torch.nn.DataParallel(self.gs, device_ids=ids).cuda()
        else:
            self.CCN = self.CCN.cuda()
            self.gs = self.gs.cuda()
        
        if cfg.LOSS_FUNCTION == 'Cos_loss':
            self.loss_mse_fn = nn.MSELoss(reduction='mean').cuda()
            self.loss_cossim_fn = CosSim_Loss().cuda()

        
        elif cfg.LOSS_FUNCTION == 'Mse_loss':
                self.loss_mse_fn = nn.MSELoss(reduction='mean').cuda()
        
        elif cfg.LOSS_FUNCTION == 'Mix_loss':
            self.loss_l1_fn = nn.L1Loss(reduction='mean').cuda()
            self.loss_mse_fn = nn.MSELoss(reduction='mean').cuda()
        
        elif cfg.LOSS_FUNCTION == 'Mask_loss':
            self.loss_l1_fn = nn.L1Loss(reduction='mean').cuda()
            self.loss_mse_fn = nn.MSELoss(reduction='mean').cuda()            
            self.loss_seg_fn = nn.CrossEntropyLoss().cuda()
        else:
            return 'error'

        
    @property
    def loss(self):
        return self.loss_fn
    
    def forward(self, data, num_class, iter = 0,  mode = 'train'):
        rgb, nir = data['rgb'], data['nir']
        nir = nir[:,0,:,:].unsqueeze(1) 
     
        if self.net_name == 'MCC':
            pred_map = self.CCN(rgb,nir)
        else:        
            pred_map = self.CCN(rgb)
            
        outputs = {
            'pred_map': pred_map}
        

        if mode == 'train':
            
       
            gt_map = data['gt_map'][:, :num_class, :, :]
            if cfg.LOSS_FUNCTION == 'Cos_loss':
                gauss_map = self.multi_class_gauss_map_generate(gt_map)
                self.loss_fn = self.build_cos_loss(pred_map, gt_map, iter, alpha=self.alpha, beta=1, lambda_=0.001, reg_params=self.CCN.parameters())
                outputs['gauss_map'] = gauss_map
                return outputs
            
            

            elif cfg.LOSS_FUNCTION == 'Mse_loss':
                gauss_map = self.multi_class_gauss_map_generate(gt_map)
                self.loss_fn = self.build_mse_loss(pred_map, gauss_map)   
                outputs['gauss_map'] = gauss_map
                return outputs
            
            elif cfg.LOSS_FUNCTION == 'Mix_loss':
               
                gauss_map = self.multi_class_gauss_map_generate(gt_map)
                self.loss_fn = self.build_mix_loss(pred_map, gt_map, iter) 
                outputs['gauss_map'] = gauss_map
                return outputs
            
            elif cfg.LOSS_FUNCTION == 'Mask_loss':
                mask_map = data['mask_map']
                pred_mask_maps = outputs['pred_mask_maps']
                gauss_map = self.multi_class_gauss_map_generate(gt_map)
                self.loss_fn = self.build_mask_loss(pred_map, gauss_map, mask_map, pred_mask_maps, iter) 
                outputs['gauss_map'] = gauss_map
                return outputs
            
            else:
                return 'error' 
        else:
            return outputs

    def build_cos_loss(self, pred_map, gauss_map, iter, alpha, beta, lambda_, reg_params):
        mse_loss = self.loss_mse_fn(pred_map, gauss_map)  
        c_loss = self.loss_cossim_fn(pred_map).cuda()


        loss = alpha * c_loss + beta * mse_loss

        if (iter+1) % 20 == 0:
            print('mse_loss:', mse_loss)
     
            print('cos_loss:', c_loss)
        return loss
   

    def build_mse_loss(self, pred_map, gauss_map):
        loss = self.loss_mse_fn(pred_map, gauss_map)  
        return loss
    
    def build_mix_loss(self, pred_map, gauss_map, iter):  
        mse_loss = self.loss_mse_fn(pred_map, gauss_map)
        l1_loss = self.loss_l1_fn(pred_map.sum(axis=[1, 2, 3]), gauss_map.sum(axis=[1, 2, 3]))
        loss = mse_loss + 1e-4*l1_loss
        if (iter+1) % 20 == 0:
            print('mse_loss:',mse_loss)
            print('l1_loss:',l1_loss)
        return loss

    def build_mask_loss(self, pred_map, gauss_map, mask_map, pred_mask_maps, iter):  
        mse_loss = self.loss_mse_fn(pred_map, gauss_map)
        l1_loss = self.loss_l1_fn(pred_map.sum(axis=[1, 2, 3]), gauss_map.sum(axis=[1, 2, 3]))
        seg_loss = 0
        for c_idx in range(mask_map.shape[1]):
            seg_loss += self.loss_seg_fn(pred_mask_maps[c_idx], mask_map[:,c_idx,:,:].long()) 
        loss = mse_loss + 1e-4 * seg_loss
        if (iter+1) % 20 == 0:
            print('mse_loss:', mse_loss)

            print('seg_loss:', seg_loss)
        return loss
    
    def multi_class_gauss_map_generate(self, gt_map):
        class_num = gt_map.shape[1]
        class_gauss_maps = []
        for i in range(class_num):
            class_gt_map = torch.unsqueeze(gt_map[:,i,:,:], 1)
            class_gauss_map = torch.squeeze(self.gs(class_gt_map), 1)
            class_gauss_maps.append(class_gauss_map)
        gauss_map = torch.stack(class_gauss_maps,1)
        return gauss_map
