import os
import math
import numpy as np
import time
import random
import shutil
import cv2
from PIL import Image

import torch
from torch import nn
import torchvision.utils as vutils
import torchvision.transforms as standard_transforms

import os.path as osp

from torch.nn import functional as F

import logging

'''
"we weight each pixel by ¦Ác = median freq/freq(c) where freq(c) is the number of pixels of class c divided by the total
number of pixels in images where c is present, and median freq is the median of these frequencies."
'''

#计算中位数
def calculate_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    return median

def softmax(x):

    x = np.array(x)
    

    exp_x = np.exp(x)
    

    softmax_x = exp_x / np.sum(exp_x)
    
    return softmax_x

def cla_weight(gt_counts):
    
    median = calculate_median(gt_counts)  
    ds = [ (value + 1.02) /  (median + 1.02) for value in gt_counts]

    
    weights = [ 1 /  np.log(d + 1.02) for d in ds]


    weights = softmax(weights)
    
    return weights

class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=None, reduction='mean')

    def forward(self, inputs, targets):
        # pdb.set_trace()
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)
    
#Mean Class MAE  &&  Meam Class MSE
def eval_mc(pred_map, gt_map, log_para):
    pred_map = pred_map.squeeze().cpu().numpy() #(num_class, 128,128)
    gt_map = gt_map.squeeze().cpu().numpy()
    class_num, H, W = gt_map.shape
    assert pred_map.shape == gt_map.shape
    weight = []
    gt_counts = []
    abs_errors = list()
    square_errors = list()   

    for i in range(class_num):


        pred_cnt = pred_map[i,:,:].sum() / log_para
        gt_count = gt_map[i,:,:].sum() / log_para
        gt_counts.append(gt_count)

        abs_error = abs(gt_count-pred_cnt)
        square_error = (gt_count-pred_cnt)*(gt_count-pred_cnt)
        abs_errors.append(abs_error)
        square_errors.append(square_error)

    weights = cla_weight(gt_counts)

    return abs_errors, square_errors, weights

def eval(pred_map, gt_map, log_para):
    pred_map = pred_map.squeeze().cpu().numpy() #(13, 128,128)
    gt_map = gt_map.squeeze().cpu().numpy()
    class_num, H, W = gt_map.shape
    assert pred_map.shape == gt_map.shape

    abs_errors = list()
    square_errors = list()    
    for i in range(class_num):
        pred_cnt = pred_map[i,:,:].sum() / log_para
        gt_count = gt_map[i,:,:].sum() / log_para
        abs_error = abs(gt_count-pred_cnt)
        square_error = (gt_count-pred_cnt)*(gt_count-pred_cnt)
        abs_errors.append(abs_error)
        square_errors.append(square_error)
    return abs_errors, square_errors


def eval_game(output, target, L=0):
    output = output.cpu().numpy() #(60, 80)
    H, W = target.shape
    assert output.shape == target.shape

    # eg: L=3, p=8 p^2=64
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    for i in range(p):
        for j in range(p):
            output_block = output[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            target_block = target[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]

            abs_error += abs(output_block.sum()-target_block.sum().float())
            square_error += (output_block.sum()-target_block.sum().float()).pow(2)

    return abs_error, square_error


def eval_relative(output, target):
    output_num = output.cpu().data.sum()
    target_num = target.sum().float()
    relative_error = abs(output_num-target_num)/target_num
    return relative_error


def initialize_weights(models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):    
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print( m )

def weights_normal_init(*models):
    for model in models:
        dev=0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():            
                if isinstance(m, nn.Conv2d):        
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)

def adjust_learning_rate(optimizer, 
                         cur_iters,
                         max_iters,
                         warmup='linear', 
                         warmup_iters=500, 
                         warmup_ratio=1e-6, 
                         lr=0.00006, 
                         power=1.0, 
                         min_lr=0.0):
    
    if warmup is not None:
        if warmup not in ['constant', 'linear', 'exp']:
            raise ValueError(
                f'"{warmup}" is not a supported type for warming up, valid'
                ' types are "constant" and "linear"')
            
    if warmup is not None:
        assert warmup_iters > 0, \
            '"warmup_iters" must be a positive integer'
        assert 0 < warmup_ratio <= 1.0, \
            '"warmup_ratio" must be in range (0,1]'
    
    if warmup == 'linear':
        k = (1-cur_iters/warmup_iters)*(1-warmup_ratio)
        warmup_lr = lr*(1-k)
    if warmup == 'constant':
        warmup_lr = lr*warmup_ratio
    if warmup == 'exp':
        k = warmup_ratio**(1-cur_iters/warmup_iters)
        warmup_lr = lr*k
    
    
    if warmup is None or  cur_iters >= warmup_iters:
        coff = (1-cur_iters/max_iters)**power
        lr = (lr - min_lr)*coff + min_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
        
        return warmup_lr

def adjust_double_learning_rate(optimizer, 
                         cur_iters,
                         max_iters,
                         policy='poly', 
                         warmup='linear', 
                         warmup_iters=1500, 
                         warmup_ratio=1e-6, 
                         base_lr=0.00006, 
                         conv_lr=0.000006, 
                         power=1.0, 
                         min_lr=0.0):
    
    if warmup is not None:
        if warmup not in ['constant', 'linear', 'exp']:
            raise ValueError(
                f'"{warmup}" is not a supported type for warming up, valid'
                ' types are "constant" and "linear"')
            
    if warmup is not None:
        assert warmup_iters > 0, \
            '"warmup_iters" must be a positive integer'
        assert 0 < warmup_ratio <= 1.0, \
            '"warmup_ratio" must be in range (0,1]'
    
    if warmup == 'linear':
        k = (1-cur_iters/warmup_iters)*(1-warmup_ratio)
        warmup_base_lr = base_lr*(1-k)
        warmup_conv_lr = conv_lr*(1-k)
    if warmup == 'constant':
        warmup_base_lr = base_lr*warmup_ratio
        warmup_conv_lr = conv_lr*warmup_ratio
    if warmup == 'exp':
        k = warmup_ratio**(1-cur_iters/warmup_iters)
        warmup_base_lr = base_lr*k
        warmup_conv_lr = conv_lr*k
    
    
    if warmup is None or  cur_iters >= warmup_iters:
        coff = (1-cur_iters/max_iters)**power
        lr1 = (base_lr - min_lr)*coff + min_lr
        lr2 = (conv_lr - min_lr)*coff + min_lr
        for i_p, param in enumerate(optimizer.param_groups,0):
            if i_p<2:
                optimizer.param_groups[i_p]['lr'] = lr2
            elif i_p<5:
                optimizer.param_groups[i_p]['lr'] = lr1
            else:
                print('Invalid lr schedule setting!')
        return lr1, lr2
    else:
        for i_p, param in enumerate(optimizer.param_groups,0):
            if i_p<2:
                optimizer.param_groups[i_p]['lr'] = warmup_conv_lr
            elif i_p<5:
                optimizer.param_groups[i_p]['lr'] = warmup_base_lr
            else:
                print('Invalid lr schedule setting!')
        return warmup_base_lr, warmup_conv_lr

def logger(exp_path, exp_name, work_dir, exception, resume=False):

    from tensorboardX import SummaryWriter
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    writer = SummaryWriter(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    if not resume:
        copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)

    return writer, log_file


def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s",
                                     "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)



def logger_txt(log_file,epoch,scores,snapshot_name, categorys):

    val_loss_avg, overall_mae, overall_mse, cls_avg_mae, cls_avg_mse, cls_weight_mse= scores
    num_classes = len(categorys)

    with open(log_file, 'a+') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')
        f.write(snapshot_name + '\n')
        for c_idx in range(num_classes):
            f.write('    %s: [mae %.2f mse %.4f]\n' % (categorys[c_idx], overall_mae[c_idx], overall_mse[c_idx]))
        f.write('  [val_loss_avg %.4f cls_avg_mae %.4f cls_avg_mse %.4f cls_weight_mse %.4f]\n' % (val_loss_avg, cls_avg_mae, cls_avg_mse, cls_weight_mse ))
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')   

def is_validation(stages,freqs,cur_epoch):

    flag = False
    stage = []

    for i_stage, cur_stage in enumerate(stages,0):
        if cur_epoch >= cur_stage:
            stage = i_stage
    
    if cur_epoch % freqs[stage] == 0:
        flag = True

    return flag
            


def vis_results(exp_name, epoch, writer, restore, img, pred_map, gt_map, factor):

    pil_to_tensor = standard_transforms.ToTensor()

    x = []
    
    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map)):
        if idx>1:# show only one group
            break

        pil_input = restore(tensor[0])
        w, h = pil_input.size

        pil_input = pil_input.resize((w//factor, h//factor), Image.BILINEAR)
        
        pred_color_map = cv2.applyColorMap((255*tensor[1]/(tensor[2].max()+1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255*tensor[2]/(tensor[2].max()+1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map,cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map,cv2.COLOR_BGR2RGB))
        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')), pil_to_tensor(pil_output.convert('RGB'))])

    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy()*255).astype(np.uint8)

    writer.add_image(exp_name + '_epoch_' + str(epoch+1), x)


def print_mm_summary(exp_name,log_txt,epoch, scores,train_record,c_maes,c_mses,c_naes):
    mae, mse, nae, loss = scores
    c_mses['level'] = np.sqrt(c_mses['level'].avg)
    c_mses['illum'] = np.sqrt(c_mses['illum'].avg)

    with open(log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(epoch) + '\n\n')

        f.write('  [mae %.4f mse %.4f nae %.4f], [val loss %.4f]\n\n' % (mae, mse, nae, loss))
        f.write('  [level: mae %.4f mse %.4f nae %.4f]\n' % (np.average(c_maes['level'].avg), np.average(c_mses['level']), np.sum(c_naes['level'].avg)/4))
        f.write('    list: ' + str(np.transpose(c_maes['level'].avg)) + '\n')
        f.write('    list: ' + str(np.transpose(c_mses['level'])) + '\n\n')
        f.write('    list: ' + str(np.transpose(c_naes['level'].avg)) + '\n\n')

        f.write('  [illum: mae %.4f mse %.4f nae %.4f]\n' % (np.average(c_maes['illum'].avg), np.average(c_mses['illum']), np.sum(c_naes['illum'].avg)/4))
        f.write('    list: ' + str(np.transpose(c_maes['illum'].avg)) + '\n')
        f.write('    list: ' + str(np.transpose(c_mses['illum'])) + '\n\n')
        f.write('    list: ' + str(np.transpose(c_naes['illum'].avg)) + '\n\n')


        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( exp_name )
    print( '    '+ '-'*20 )
    print( '    [mae %.2f mse %.2f], [val loss %.4f]' % (mae, mse, loss) )
    print( '    '+ '-'*20 )
    print( '[best] [model: %s] , [mae %.2f], [mse %.2f], [nae %.4f]' % (train_record['best_model_name'],\
                                                        train_record['best_mae'],\
                                                        train_record['best_mse'],\
                                                        train_record['best_nae']) )
    print( '='*50 )  


# def update_model(net,optimizer,epoch,i_tb,exp_path,exp_name,scores,train_record,log_file=None):

#     mae, mse, nae, loss = scores

#     # import pdb
#     # pdb.set_trace()

#     snapshot_name = 'all_ep_%d_mae_%.1f_mse_%.1f_nae_%.3f' % (epoch + 1, mae, mse, nae)

#     if mae < train_record['best_mae'] or mse < train_record['best_mse'] or nae < train_record['best_nae']:   
#         train_record['best_model_name'] = snapshot_name
#         if log_file is not None:
#             logger_txt(log_file,epoch,scores)
#         to_saved_weight = net.state_dict()
#         torch.save(to_saved_weight, os.path.join(exp_path, exp_name, snapshot_name + '.pth'))

#     if mae < train_record['best_mae']:           
#         train_record['best_mae'] = mae
#     if mse < train_record['best_mse']:
#         train_record['best_mse'] = mse 
#     if nae < train_record['best_nae']:
#         train_record['best_nae'] = nae 

#     return train_record

def update_model(net, optimizer, epoch, i_tb, exp_path, exp_name, scores, train_record, log_file, categorys):

    val_loss_avg, overall_mae, overall_mse, cls_avg_mae, cls_avg_mse, cls_weight_mse = scores

    # import pdb
    # pdb.set_trace()
    snapshot_name = 'all_ep_%d_cls_avg_mae_%.1f_cls_avg_mse_%.1f_cls_weight_mse_%.1f' % (epoch + 1, cls_avg_mae, cls_avg_mse, cls_weight_mse)

    if cls_avg_mae < train_record['best_cls_avg_mae'] or cls_avg_mse < train_record['best_cls_avg_mse'] or cls_weight_mse < train_record['best_cls_weight_mse']:   
        train_record['best_model_name'] = snapshot_name
        if log_file is not None:
            logger_txt(log_file, epoch, scores, snapshot_name, categorys)
        to_saved_weight = net.state_dict()
        torch.save(to_saved_weight, os.path.join(exp_path, exp_name, snapshot_name + '.pth'))

    if cls_avg_mae < train_record['best_cls_avg_mae']:           
        train_record['best_cls_avg_mae'] = cls_avg_mae
        train_record['overall_mae'] = overall_mae
    if cls_avg_mse < train_record['best_cls_avg_mse']:           
        train_record['best_cls_avg_mse'] = cls_avg_mse
        train_record['overall_mae'] = overall_mse
        
    if cls_weight_mse < train_record['best_cls_weight_mse']:           
        train_record['best_cls_weight_mse'] = cls_weight_mse
        train_record['overall_mae'] = overall_mse
    return train_record


def save_checkpoint(trainer):

    latest_state = {'train_record':trainer.train_record, 'net':trainer.net.state_dict(), 'optimizer':trainer.optimizer.state_dict(),\
                     'epoch': trainer.epoch, 'i_tb':trainer.i_tb, 'num_iters':trainer.num_iters, 'exp_path':trainer.exp_path, \
                    'exp_name':trainer.exp_name}

    torch.save(latest_state,os.path.join(trainer.exp_path, trainer.exp_name, 'latest_state.pth'))

def copy_cur_env(work_dir, dst_dir, exception):

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)

        if os.path.isdir(file) and exception not in filename:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count
        
class AveragePosNegMeter(object):
    """按照正负样本分别计算MAE & MSE"""

    def __init__(self,num_class):        
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.pn_avg = np.zeros(self.num_class)
        
        self.pos_avg = np.zeros(self.num_class)
        self.neg_avg = np.zeros(self.num_class)
        self.pos_sum = np.zeros(self.num_class)
        self.neg_sum = np.zeros(self.num_class)
        self.pos_count = np.zeros(self.num_class)
        self.neg_count = np.zeros(self.num_class)

    def update(self, cur_val, pos, class_id):
        self.cur_val[class_id] = cur_val
        if pos == 1:          #正样本
            self.pos_count[class_id] += 1
            self.pos_sum[class_id] += cur_val
        else :
            self.neg_count[class_id] += 1
            self.neg_sum[class_id] += cur_val
            

        self.pos_avg[class_id] = self.pos_sum[class_id] / self.pos_count[class_id]
        self.neg_avg[class_id] = self.neg_sum[class_id] / self.neg_count[class_id]
        self.pn_avg[class_id] = ( self.pos_avg[class_id] + self.neg_avg[class_id] ) / 2 


class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self,num_class):        
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.avg = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)
        self.count = np.zeros(self.num_class)

    def update(self, cur_val, class_id):
        self.cur_val[class_id] = cur_val
        self.sum[class_id] += cur_val
        self.count[class_id] += 1
        self.avg[class_id] = self.sum[class_id] / self.count[class_id]



class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    rgb_list = transposed_batch[0]
    ir_list = transposed_batch[1]
    gt_list = transposed_batch[2]
    # mask_list = transposed_batch[3]
    rgb = torch.stack(rgb_list, 0)
    ir = torch.stack(ir_list, 0)
    gt_map = torch.stack(gt_list, 0)
    # mask_map = torch.stack(mask_list, 0)
    data = dict()
    data['rgb'] = rgb
    data['nir'] = ir
    data['gt_map'] = gt_map
    # data['mask_map'] = mask_map
    return data