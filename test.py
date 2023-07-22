from locale import getlocale
import os
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
from importlib import import_module

from models.CC import CrowdCounter

from misc.utils import *

from misc import layer
from PIL import Image

import time

from torch.utils.tensorboard import SummaryWriter

import datasets

from config import cfg

seed = 42
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

#------------prepare data loader------------
dataset = 'MOC_RS'

model_path = ''

datasetting = import_module(f'datasets.setting.{dataset}')
cfg_data = datasetting.cfg_data

net_name = 'Res101_SFCN'
gpu_id = [0]

if len(gpu_id)==1:
    torch.cuda.set_device(gpu_id[0])
torch.backends.cudnn.benchmark = True

visible = True
vis_freq = 1 
attn_visible = False
attn_vis_freq = 20 

pos_embedding = False

categorys = cfg_data.CATEGORYS
log_para = cfg_data.LOG_PARA
label_factor = cfg_data.LABEL_FACTOR

test_size = cfg.TRAIN_SIZE

mean_std_rgb = cfg_data.MEAN_STD_RGB
mean_std_ir = cfg_data.MEAN_STD_IR

now = time.strftime("%m-%d_%H-%M", time.localtime())
test_name = now \
            + '_' + model_path.replace("/","_").replace(".pth","")
    
save_path = os.path.join('exp_test',test_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
writer = SummaryWriter(save_path)   

def main():
    test_loader = datasets.loading_test_data(dataset)
    net = CrowdCounter(net_name, gpu_id)
    net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpu_id[0])), strict=True)                     
                       
    print("__loading__test__model__")
    net.eval()
    test(net, test_loader)

restore_transform_rgb = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std_rgb),
        standard_transforms.ToPILImage()
    ])

restore_transform_t = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std_ir),
        standard_transforms.ToPILImage()
    ])

   


def multi_class_gauss_map_generate(gt_map):
    gs_layer = getattr(layer, 'Gaussianlayer')
    gs = gs_layer()
    if len(gpu_id) > 1:
        gs = torch.nn.DataParallel(gs, device_ids=gpu_id).cuda()
    else:
        gs = gs.cuda()
    class_gauss_maps = []
    for i in range(len(categorys)):
        class_gt_map = torch.unsqueeze(gt_map[:,i,:,:], 1)
        class_gauss_map = torch.squeeze(gs(class_gt_map), 1)
        class_gauss_maps.append(class_gauss_map)
    gauss_map = torch.stack(class_gauss_maps,1)
    return gauss_map

def save_counting_results(pred_map, gt_map, record, index):
    pred_map = pred_map.data.cpu().numpy()
    gt_map = gt_map.data.cpu().numpy()
    
    for c_idx in range(len(categorys)):
        pred_cnt = np.sum(pred_map[:,c_idx,:,:])/log_para
        gt_count = np.sum(gt_map[:,c_idx,:,:])/log_para
        print(categorys[c_idx], '   [cnt: gt: %.1f pred: %.2f]' % (gt_count, pred_cnt) )          
        print(f'{index} {categorys[c_idx]} {gt_count:.1f} {pred_cnt:.4f}', file=record)
        
def test(net, test_loader):
    
    print('testing...')
    # Iterate over data.
    maes = AverageCategoryMeter(len(categorys))
    mses = AverageCategoryMeter(len(categorys))
    cmses = AverageMeter()
    

    record_path = os.path.join(save_path, 'record_results.txt')
    record = open(record_path, 'w+')
    
    num_class = 6
    
    print(model_path)
    print(model_path, file=record)
    
    for index, data in enumerate(test_loader, 0):
        # get_local.clear()
        print("test:",index,"/ ", len(test_loader))
        for k, v in data.items(): # data to cuda
            data[k] = v.cuda()
        rgb, nir, gt_map = data['rgb'], data['nir'], data['gt_map'][:, :num_class, :, :]
        
        gauss_map = multi_class_gauss_map_generate(gt_map)

        with torch.set_grad_enabled(False):
            if pos_embedding:
                b, c, h, w = rgb.shape
                rh, rw = test_size

                crop_RGBs, crop_Ts, crop_masks = [],[],[]

                for i in range(0, h, rh): # i=0,256
                    # (0,256),(224,480),crop blcok overlapping
                    gis, gie = max(min(h-rh, i), 0), min(h, i+rh) 
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w-rw, j), 0), min(w, j+rw)
                        crop_RGBs.append(rgb[:, :, gis:gie, gjs:gje])
                        crop_Ts.append(nir[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros(b, 1, h//label_factor, w//label_factor).cuda()
                        mask[:, :, gis//label_factor:gie//label_factor, gjs//label_factor:gje//label_factor].fill_(1.0)
                        crop_masks.append(mask)

                crop_RGBs, crop_Ts, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_RGBs, crop_Ts, crop_masks))

                crop_data = {'rgb': crop_RGBs, 'nir': crop_Ts,}
                crop_outputs = net(crop_data, num_class, mode = 'test')
                crop_preds = crop_outputs['pred_map']

                h, w, rh, rw = h//label_factor, w//label_factor, rh//label_factor, rh//label_factor

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
                outputs = net(data, num_class, mode = 'test')
                pred_map = outputs['pred_map']            

            abs_errors, square_errors, weights = eval_mc(pred_map, gt_map, log_para)

            
            wmse = 0.0
            for c_idx in range(len(categorys)):
                maes.update(abs_errors[c_idx], c_idx)
                mses.update(square_errors[c_idx], c_idx)                         
                wmse += square_errors[c_idx] * weights[c_idx]

            cmses.update(wmse)
               
            save_counting_results(pred_map, gt_map, record, index)


    N = len(test_loader)
    overall_mae = maes.avg # list
    overall_rmse = np.sqrt(mses.avg)
    cls_weight_mse = cmses.avg

    
    cls_avg_mae = sum(overall_mae) / len(categorys)
    cls_avg_rmse = sum(overall_rmse) / len(categorys)


    log_str = 'Test{}, cls_avg_mae {cls_avg_mae:.4f} cls_avg_mse {cls_avg_rmse:.4f} cls_weight_mse {cls_weight_mse:.4f}'.\
            format(N, cls_avg_mae=cls_avg_mae, cls_avg_rmse=cls_avg_rmse , cls_weight_mse=cls_weight_mse)

    print(log_str)
    print(log_str, file=record)

    for c_idx in range(len(categorys)):
        class_str = '{category}, mae: {mae:.4f} rmse: {rmse:.4f}'\
                .format(category = categorys[c_idx], mae = overall_mae[c_idx], rmse = overall_rmse[c_idx])
        print(class_str)
        print(class_str, file=record)   
    
    print(model_path)
    print(model_path, file=record)
    
    record.close()


if __name__ == '__main__':
    main()
