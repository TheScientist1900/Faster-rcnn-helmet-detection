from yacs.config import CfgNode as CN
import os

from utils.utils import get_classes

_C = CN()
_C.num_workers = 0
_C.pin_memory = True
_C.cuda = True
_C.gpus = [0]

_C.arch = 'vgg' # backbone
_C.data_name = ''   # VOCdevkit路径下数据集存放的文件夹的名字，在这里默认会是VOC2028
_C.checkpoint = ''  # 需要加载的预训练模型的相对路径
_C.pretrained = False   # backbone是否加载从Imagenet上训练好的模型

_C.cudnn = CN()
_C.cudnn.benchmark = True
_C.cudnn.deterministic = True
_C.cudnn.enabled = True

# D:\faster-rcnn-pytorch-master\utils\..
_C.root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..',)
_C.data = CN()
_C.data.name = 'VOC2028'
_C.data.root = os.path.join(_C.root, 'VOCdevkit', _C.data.name)
_C.data.input_shape = [600, 600]
_C.data.num_classes = get_classes(
    os.path.join(_C.data.root, 'voc_classes.txt'))[1] + 1
_C.data.class_names = ['__background__'] + \
    get_classes(os.path.join(_C.data.root, 'voc_classes.txt'))[0]

_C.model = CN()

_C.model.rpn = CN()
_C.model.rpn.mid_channels = 512
_C.model.rpn.scales = [1, 4, 8, 16, 32, 128, 256, 512]  # 初始的anchor_size
# _C.model.rpn.scales = [8,16,32]

_C.model.rpn.ratios = [0.5, 1, 2]
_C.model.rpn.feat_stride = 16
_C.model.rpn.base_size = 1
_C.model.rpn.proposal_creator = CN()
_C.model.rpn.proposal_creator.n_train_pre_nms = 6000
_C.model.rpn.proposal_creator.n_train_post_nms = 600
_C.model.rpn.proposal_creator.n_test_pre_nms = 3000
_C.model.rpn.proposal_creator.n_test_post_nms = 300
_C.model.rpn.proposal_creator.nms_iou = 0.7
_C.model.rpn.proposal_creator.min_size = 2

_C.model.roi_head = CN()
_C.model.roi_head.roi_size = 7
_C.model.roi_head.spatial_scale = 1

_C.train = CN()
_C.train.pretrained = True
_C.train.batch_size = 16
_C.train.shuffle = True

_C.train.freeze = True  # 是否进行freeze train, 即是否先冻结backbone的参数训练几轮后再解冻训练整个网络的参数
_C.train.freeze_lr = 1e-5   # 冻结训练时的lr
_C.train.unfreeze_lr = 1e-4 # 解冻训练时的lr
_C.train.wd = 5e-4
_C.train.freeze_epoch = 15
_C.train.epoch = 30 # 总共训练的轮数

_C.train.anchor_target_creator = CN()
_C.train.anchor_target_creator.n_sample = 256
_C.train.anchor_target_creator.pos_ratio = 0.5
_C.train.anchor_target_creator.pos_iou_thresh = 0.7
_C.train.anchor_target_creator.neg_iou_thresh = 0.3

_C.train.proposal_target_creator = CN()
_C.train.proposal_target_creator.n_sample = 128
_C.train.proposal_target_creator.pos_ratio = 0.5
_C.train.proposal_target_creator.pos_iou_thresh = 0.7
_C.train.proposal_target_creator.neg_iou_thresh_high = 0.5
_C.train.proposal_target_creator.neg_iou_thresh_low = 0
_C.train.proposal_target_creator.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

_C.test = CN()
_C.test.confidence = 0.5
_C.test.nms_iou = 0.3
_C.test.loc_std = [0.1, 0.1, 0.2, 0.2]

_C.test.map = CN()
_C.test.map.map_out_dir = os.path.join(_C.root, 'map_out')
_C.test.map.map_vis = True

# 1 predict - 2 gt -  3 get map
# 0 1 || 2 || 3
_C.test.map.mode = 0
_C.test.map.data_file = os.path.join(
    _C.root, 'VOCdevkit', _C.data.name, 'ImageSets/Main/test.txt')


def update_config(cfg, arg):
    cfg.defrost()
    arg_list = []
    for (key, item) in vars(arg).items():
        if key == 'data_name':
            cfg.data.name = item
            cfg.data.root = os.path.join(cfg.root, 'VOCdevkit', cfg.data.name)
            cfg.test.map.data_file = os.path.join(
                _C.root, 'VOCdevkit', _C.data.name, 'test.txt')

        arg_list.append(key)
        arg_list.append(item)
    cfg.defrost()
    cfg.merge_from_list(arg_list)
    cfg.freeze()
    return cfg
