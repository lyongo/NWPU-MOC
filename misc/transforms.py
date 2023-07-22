import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from config import cfg
import torch
from torchvision.transforms import functional as TrF
# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, nir, mask, bbx=None):
        if bbx is None:
            for trans in self.transforms:
                rgb, nir, mask = trans(rgb, nir, mask)
            return rgb, nir, mask
        for trans in self.transforms:
            rgb, nir, mask, bbx = trans(rgb, nir, mask, bbx)
        return rgb, nir, mask, bbx

class RandomHorizontallyFlip(object):
    def __call__(self, rgb, ir, gt_map, bbx=None):
        if random.random() < 0.5:
            if bbx is None:
                return rgb.transpose(Image.FLIP_LEFT_RIGHT), ir.transpose(Image.FLIP_LEFT_RIGHT), gt_map[:,::-1,:]
            w, h = rgb.size
            xmin = w - bbx[:,3]
            xmax = w - bbx[:,1]
            bbx[:,1] = xmin
            bbx[:,3] = xmax
            return rgb.transpose(Image.FLIP_LEFT_RIGHT), ir.transpose(Image.FLIP_LEFT_RIGHT), gt_map[:,::-1,:], bbx
        if bbx is None:
            return rgb, ir, gt_map, 
        return rgb, ir, gt_map,  bbx

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, rgb, nir, gt_map, dst_size=None):
        if self.padding > 0:
            rgb = ImageOps.expand(rgb, border=self.padding, fill=0)
            nir = ImageOps.expand(nir, border=self.padding, fill=0)
            gt_map = np.pad(gt_map, pad_width=self.padding, mode='constant')
            # mask_map = np.pad(mask_map, pad_width=self.padding, mode='constant')
        
        w, h = rgb.size
        if dst_size is None:
            th, tw = self.size
        else:
            th, tw = dst_size
        if w == tw and h == th:
            return rgb, nir, gt_map

        assert w >= tw
        assert h >= th

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return rgb.crop((x1, y1, x1 + tw, y1 + th)), nir.crop((x1, y1, x1 + tw, y1 + th)), \
                gt_map[y1:y1 + th, x1:x1 + tw, :]

class ScaleDown(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, mask):
        return  mask.resize((self.size[1]/cfg.TRAIN.DOWNRATE, self.size[0]/cfg.TRAIN.DOWNRATE), Image.NEAREST)

class ScaleByRateWithMin(object):
    def __init__(self, rateRange, min_h, min_w):
        self.rateRange = rateRange
        self.min_h = min_h
        self.min_w = min_w
        
    def __call__(self, rgb, nir, gt_map):
        h, w, class_num = gt_map.shape

        rate = random.uniform(self.rateRange[0], self.rateRange[1])
        new_h = int(h * rate) // 32 * 32
        new_w = int(w * rate) // 32 * 32
        
        if new_h< self.min_h or new_w<self.min_w:
            if new_w<self.min_w:
                new_w = self.min_w
                rate = new_w/w
                new_h = int(h*rate) // 32*32
            if new_h < self.min_h:
                new_h = self.min_h
                rate = new_h / h
                new_w =int( w * rate) //32*32
                
        new_gt_map = np.zeros((new_h, new_w, class_num)).astype(np.float32)
        for i in range(class_num):
            ori_points = np.argwhere(gt_map[:,:,i] >= 1)
            points = ori_points.copy()                                                                    
            points[:, 0] = (points[:, 0] / h) * new_h
            points[:, 0] = [int(x) for x in points[:, 0]]
            points[:, 1] = (points[:, 1] / w) * new_w
            points[:, 1] = [int(x) for x in points[:, 1]]
            for point, ori_point in zip(points, ori_points): 
                new_gt_map[point[0], point[1], i] += gt_map[ori_point[0], ori_point[1], i]
            assert new_gt_map[:,:,i].sum() == gt_map[:,:,i].sum(), \
                "new_gt_map:{}, gt_map:{}".format(new_gt_map[:,:,i].sum(), gt_map[:,:,i].sum())            

        # new_mask_map = np.zeros((new_h, new_w, class_num)).astype(np.float32)
        # for i in range(class_num):
        #     # ori_points = np.argwhere(mask_map[:,:,i] > 0)
        #     points = ori_points.copy()
        #     points[:, 0] = (points[:, 0] / h) * new_h
        #     points[:, 0] = [int(x) for x in points[:, 0]]
        #     points[:, 1] = (points[:, 1] / w) * new_w
        #     points[:, 1] = [int(x) for x in points[:, 1]]
            # for point in points: 
            #     if new_mask_map[point[0], point[1], i] == 0:
            #         new_mask_map[point[0], point[1], i] = 1
        
        rgb = rgb.resize((new_h, new_w), Image.BILINEAR)
        nir = nir.resize((new_h, new_w), Image.BILINEAR)
        return rgb, nir, new_gt_map     


# ===============================image tranforms============================

class RGB2Gray(object):
    def __init__(self, ratio, output_channel):
        self.ratio = ratio  # [0-1]
        self.output_channel = output_channel

    def __call__(self, img):
        if random.random() < 0.1:
            return  TrF.to_grayscale(img, num_output_channels=self.output_channel)
        else: 
            return img

class GammaCorrection(object):
    def __init__(self, gamma_range=[0.4,2]):
        self.gamma_range = gamma_range 

    def __call__(self, img):
        if random.random() < 0.5:
            gamma = random.uniform(self.gamma_range[0],self.gamma_range[1])
            return  TrF.adjust_gamma(img, gamma)
        else: 
            return img

# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for nir, m, s in zip(tensor, self.mean, self.std):
            nir.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor*self.para
        return tensor

class GTScaleDownOld(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor==1:
            return img
        tmp = np.array(img.resize((w//self.factor, h//self.factor), Image.BICUBIC))*self.factor*self.factor
        img = Image.fromarray(tmp)
        return img

class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor
    def __call__(self, gt_map):
        h, w, class_num = gt_map.shape
        if self.factor==1:
            return gt_map

        new_h = h // self.factor
        new_w = w // self.factor

        new_gt_map = np.zeros((new_h, new_w, class_num)).astype(np.float32)
        for i in range(class_num):
            ori_points = np.argwhere(gt_map[:,:,i] >= 1)
            points = ori_points.copy()
            points[:, 0] = (points[:, 0] / h) * new_h
            points[:, 0] = [int(x) for x in points[:, 0]]
            points[:, 1] = (points[:, 1] / w) * new_w
            points[:, 1] = [int(x) for x in points[:, 1]]
            for point, ori_point in zip(points, ori_points): 
                new_gt_map[point[0], point[1], i] += gt_map[ori_point[0], ori_point[1], i]
            assert new_gt_map[:,:,i].sum() == gt_map[:,:,i].sum(), \
                "new_gt_map:{}, gt_map:{}".format(new_gt_map[:,:,i].sum(), gt_map[:,:,i].sum())

        return new_gt_map

class MaskScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor
    def __call__(self, mask_map):
        h, w, class_num = mask_map.shape
        if self.factor==1:
            return mask_map

        new_h = h // self.factor
        new_w = w // self.factor

        new_mask_map = np.zeros((new_h, new_w, class_num)).astype(np.float32)
        for i in range(class_num):
            ori_points = np.argwhere(mask_map[:,:,i] > 0)
            points = ori_points.copy()
            points[:, 0] = (points[:, 0] / h) * new_h
            points[:, 0] = [int(x) for x in points[:, 0]]
            points[:, 1] = (points[:, 1] / w) * new_w
            points[:, 1] = [int(x) for x in points[:, 1]]
            for point in points: 
                if new_mask_map[point[0], point[1], i] == 0:
                    new_mask_map[point[0], point[1], i] = 1
        return new_mask_map


class tensormul(object):
    def __init__(self, mu=255.0):
        self.mu = 255.0
    
    def __call__(self, _tensor):
        _tensor.mul_(self.mu)
        return _tensor
