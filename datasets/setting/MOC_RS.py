from easydict import EasyDict as edict
from config import cfg

# init
__C_MOC_RS = edict()

cfg_data = __C_MOC_RS

__C_MOC_RS.TRAIN_SIZE = cfg.TRAIN_SIZE
__C_MOC_RS.DATA_PATH = 'DataSet/MOC_Pre'
__C_MOC_RS.TRAIN_LST = 'train.txt'
__C_MOC_RS.VAL_LST =  'val.txt'
__C_MOC_RS.TEST_LST =  'test.txt'

__C_MOC_RS.CATEGORYS = ['ship', 'vehicle', 'building', 'container', 'tree', 'airplane'] 

__C_MOC_RS.MEAN_STD_RGB = ([0.446722, 0.445974, 0.425395], [0.171309, 0.147749, 0.134108])
__C_MOC_RS.MEAN_STD_IR = ([0.532902, 0.447160, 0.445926], [0.197593, 0.170431, 0.146572])

__C_MOC_RS.LABEL_FACTOR = 8 # label downsample_ratio
__C_MOC_RS.LOG_PARA = 100. 

__C_MOC_RS.RESUME_MODEL = ''  #model path
__C_MOC_RS.TRAIN_BATCH_SIZE = cfg.TRAIN_BATCH_SIZE #imgs

__C_MOC_RS.VAL_BATCH_SIZE = 1 # must be 1
