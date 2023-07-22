import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import datasets

from tqdm import tqdm
from torchvision.utils import make_grid
from misc.utils import adjust_learning_rate, adjust_double_learning_rate

class Trainer():
    def __init__(self, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.train_loader, self.val_loader = datasets.loading_data(cfg.DATASET)

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET
        self.gpu_id = cfg.GPU_ID

        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME) 

        self.net = CrowdCounter(self.net_name, self.gpu_id)
        
        self.categorys = self.cfg_data.CATEGORYS
        self.num_classes = len(self.categorys)

        if cfg.BACKBONE_FREEZE:       
            self.optimizer = optim.AdamW([
                {'params': [param for name, param in self.net.named_parameters() if 'backbone' in name], 'lr': cfg.CONV_LR, 'weight_decay': cfg.WEIGHT_DECAY},
                {'params': [param for name, param in self.net.named_parameters() if 'backbone' not in name], 'lr': cfg.BASE_LR, 'weight_decay': cfg.WEIGHT_DECAY},
            ])
        else:
            self.optimizer = optim.AdamW(self.net.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)

        self.train_record = {'best_cls_avg_mae': 1e20, 'best_cls_avg_mse':1e20, 'best_cls_weight_mse':1e20, 'best_model_name': ''}
        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

        self.epoch = 0
        self.i_tb = 0
        self.num_iters = cfg.MAX_EPOCH * np.int(len(self.train_loader))
        self.train_loss_avg = 0
        self.val_loss_avg = 0
        
        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.num_iters = latest_state['num_iters']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']


    def forward(self):

        # self.validate()
        for epoch in range(self.epoch,cfg.MAX_EPOCH):
            self.epoch = epoch
            record_path = os.path.join(self.exp_path, self.exp_name, 'log.txt')
            record = open(record_path, 'a+')

            # training    
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff), file=record )
            if self.epoch % cfg.CP_FREQ == 0:
                save_checkpoint(self)
            print( '='*20 )
            print( '='*20, file=record )

            self.writer.add_scalar('train_loss_avg', self.train_loss_avg, self.epoch)
            self.train_loss_avg = 0
            
            # validation
            if is_validation(cfg.VAL_STAGE,cfg.VAL_FREQ,self.epoch):
                self.timer['val time'].tic()
                self.validate() # only when pos_emb = False
                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff), file=record  )
                self.writer.add_scalar('val_loss_avg', self.val_loss_avg, self.epoch)
                self.val_loss_avg = 0
          
            record.close()
            
    def train(self): # training for all datasets
        
        self.net.train()
        
        train_losses = AverageMeter()
        
        for it, data in enumerate(self.train_loader, 0):
            self.i_tb += 1
            self.timer['iter time'].tic()
            for k, v in data.items(): # data to cuda
                data[k] = v.cuda()

            num_class = self.num_classes
            rgb, nir, gt_map = data['rgb'], data['nir'], data['gt_map'][:, :num_class, :, :]

            self.optimizer.zero_grad()

            # input_grid = make_grid([rgb[0],nir[0]], nrow=2, normalize=True, scale_each=True)
            # self.writer.add_image("inputs", input_grid, it)

            outputs = self.net(data, num_class, it)
            pred_map = outputs['pred_map']
            gauss_map = outputs['gauss_map']
            
            
            output_list = list()
            # for c_idx in range(self.num_classes):
            #     output_list.extend([gt_map[0][c_idx].unsqueeze(dim=0), \
            #                         gauss_map[0][c_idx].unsqueeze(dim=0), \
            #                         pred_map[0][c_idx].unsqueeze(dim=0)])
            # output_grid = make_grid(output_list, nrow=4, normalize=True, scale_each=True)#排列图像
            # self.writer.add_image("outputs", output_grid, it)

            loss = self.net.loss
            loss.backward()
            self.optimizer.step()
            
            train_losses.update(loss.item())
            
            if cfg.BACKBONE_FREEZE:
                base_lr, conv_lr = adjust_double_learning_rate(self.optimizer, self.i_tb, self.num_iters, base_lr=cfg.BASE_LR, conv_lr = cfg.CONV_LR)
            else:
                base_lr = adjust_learning_rate(self.optimizer, self.i_tb, self.num_iters, lr=cfg.BASE_LR)
                
            record_path = os.path.join(self.exp_path, self.exp_name, 'log.txt')
            record = open(record_path, 'a+')

            if (it + 1) % cfg.PRINT_FREQ == 0:
                self.writer.add_scalar('train_base_lr', base_lr, self.i_tb)
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)

                if cfg.BACKBONE_FREEZE:
                    print( '[ep %d][it %d][loss %.4f][conv_lr %.8f][base_lr %.8f][%.2fs]' % \
                            (self.epoch + 1, it + 1, loss.item(), self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr'], self.timer['iter time'].diff) )
                    print( '[ep %d][it %d][loss %.4f][conv_lr %.8f][base_lr %.8f][%.2fs]' % \
                            (self.epoch + 1, it + 1, loss.item(), self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr'], self.timer['iter time'].diff), file=record )
                
                else:      
                    print( '[ep %d][it %d][loss %.4f][base_lr %.8f][%.2fs]' % \
                            (self.epoch + 1, it + 1, loss.item(), self.optimizer.param_groups[0]['lr'], self.timer['iter time'].diff) )
                    print( '[ep %d][it %d][loss %.4f][base_lr %.8f][%.2fs]' % \
                            (self.epoch + 1, it + 1, loss.item(), self.optimizer.param_groups[0]['lr'], self.timer['iter time'].diff), file=record )
                for c_idx in range(self.num_classes):
                    print( '       ', self.categorys[c_idx],':  [cnt: gt: %.1f pred: %.2f]' % (gt_map[0][c_idx].sum().data/self.cfg_data.LOG_PARA, pred_map[0][c_idx].sum().data/self.cfg_data.LOG_PARA) ) 
                    print( '       ', self.categorys[c_idx],':  [cnt: gt: %.1f pred: %.2f]' % (gt_map[0][c_idx].sum().data/self.cfg_data.LOG_PARA, pred_map[0][c_idx].sum().data/self.cfg_data.LOG_PARA), file=record ) 
            record.close()

        self.train_loss_avg = train_losses.avg

    def validate(self): 
        
        self.net.eval()  


        val_losses = AverageMeter()
        maes = AverageCategoryMeter(self.num_classes)
        mses = AverageCategoryMeter(self.num_classes)
        cmses = AverageMeter()


        for index, data in enumerate(self.val_loader, 0):
            for k, v in data.items(): # data to cuda
                data[k] = v.cuda()
            
            num_class = self.num_classes    
            rgb, nir, gt_map = data['rgb'], data['nir'], data['gt_map'][:, :num_class, :, :]

            with torch.set_grad_enabled(False):
                if cfg.POS_EMBEDDING:
                    b, c, h, w = rgb.shape
                    rh, rw = self.cfg_data.TRAIN_SIZE

                    crop_RGBs, crop_Ts, crop_masks = [],[],[] # mask是用来记录overlapping的部分重叠过几次的

                    for i in range(0, h, rh): # i=0,256
                        # (0,256),(224,480),crop blcok overlapping
                        gis, gie = max(min(h-rh, i), 0), min(h, i+rh) 
                        for j in range(0, w, rw):
                            gjs, gje = max(min(w-rw, j), 0), min(w, j+rw)
                            crop_RGBs.append(rgb[:, :, gis:gie, gjs:gje])
                            crop_Ts.append(nir[:, :, gis:gie, gjs:gje])
                            mask = torch.zeros(b, 1, h//self.cfg_data.LABEL_FACTOR, w//self.cfg_data.LABEL_FACTOR).cuda()
                            mask[:, :, gis//self.cfg_data.LABEL_FACTOR:gie//self.cfg_data.LABEL_FACTOR, gjs//self.cfg_data.LABEL_FACTOR:gje//self.cfg_data.LABEL_FACTOR].fill_(1.0)
                            crop_masks.append(mask)

                    crop_RGBs, crop_Ts, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_RGBs, crop_Ts, crop_masks))

                    crop_data = {'rgb': crop_RGBs, 'nir': crop_Ts,}
                    crop_outputs = self.net(crop_data, mode = 'val')
                    crop_preds = crop_outputs['pred_map']

                    h, w, rh, rw = h//self.cfg_data.LABEL_FACTOR, w//self.cfg_data.LABEL_FACTOR, rh//self.cfg_data.LABEL_FACTOR, rh//self.cfg_data.LABEL_FACTOR

                    idx = 0
                    pred_map = torch.zeros(b, 1, h, w).cuda()
                    for i in range(0, h, rh):
                        gis, gie = max(min(h-rh, i), 0), min(h, i+rh)
                        for j in range(0, w, rw):
                            gjs, gje = max(min(w-rw, j), 0), min(w, j+rw)
                            pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                            idx += 1

                    mask = crop_masks.sum(dim=0)
                    pred_map = pred_map / mask

                else:
                    outputs = self.net(data, num_class, mode = 'val')
                    pred_map = outputs['pred_map']

                val_losses.update(self.net.loss.item())
                
                # abs_errors, square_errors = eval(pred_map, gt_map, self.cfg_data.LOG_PARA)
                abs_errors, square_errors, weights = eval_mc(pred_map, gt_map, self.cfg_data.LOG_PARA)
                wmse = 0.0
                for c_idx in range(self.num_classes):
                    maes.update(abs_errors[c_idx], c_idx)
                    mses.update(square_errors[c_idx], c_idx)
                    wmse += square_errors[c_idx] * weights[c_idx]


                cmses.update(wmse)
                
        N = len(self.val_loader)                 
        self.val_loss_avg = val_losses.avg
        overall_mae = maes.avg # list
        # RMSE ??
        overall_rmse = np.sqrt(mses.avg) 
        cls_weight_mse = cmses.avg
        
        cls_avg_mae = sum(overall_mae) / self.num_classes
        cls_avg_rmse = sum(overall_rmse) / self.num_classes

        self.writer.add_scalar('val_loss', self.val_loss_avg, self.epoch + 1)
        self.writer.add_scalar('cls_weight_mse', cls_weight_mse, self.epoch + 1)
        self.writer.add_scalar('cls_avg_mae', cls_avg_mae, self.epoch + 1)
        self.writer.add_scalar('cls_avg_rmse', cls_avg_rmse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [self.val_loss_avg, overall_mae, overall_rmse, cls_avg_mae, cls_avg_rmse, cls_weight_mse], self.train_record, self.log_txt, self.categorys)

        record_path = os.path.join(self.exp_path, self.exp_name, 'log.txt')
        record = open(record_path, 'a+')
        
        log_str = 'Val{}, val_loss {val_loss:.4f} cls_avg_mae {cls_avg_mae:.4f} cls_avg_rmse {cls_avg_rmse:.4f} cls_weight_mse {cls_weight_mse:.4f}'.\
                format(N, val_loss=self.val_loss_avg, cls_avg_mae=cls_avg_mae, cls_avg_rmse=cls_avg_rmse,  cls_weight_mse=cls_weight_mse)
        
        print(log_str)
        print(log_str, file=record)
        
        for c_idx in range(self.num_classes):

          
            class_str = '{category}, mae: {mae:.4f} rmse: {rmse:.4f}'\
                .format(category = self.categorys[c_idx], mae = overall_mae[c_idx], rmse = overall_rmse[c_idx])
            print(class_str)
            print(class_str, file=record)   
        
        record.close()
