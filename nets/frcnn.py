import torch.nn as nn

from nets.classifier import ResnetRoIHead, VGG16RoIHead
from nets.resnet import resnet
from nets.rpn import RegionProposalNetwork
from nets.vgg16 import decom_vgg16


class FasterRCNN(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(FasterRCNN, self).__init__()
        #---------------------------------#
        #   一共存在两个主干
        #   vgg和resnet50
        #---------------------------------#
        if cfg.arch.__contains__('vgg'):
            self.extractor, classifier = decom_vgg16(pretrained=cfg.pretrained)
            #---------------------------------#
            #   构建建议框网络
            #---------------------------------#
            in_channels = list(self.extractor.children())[-2].in_channels
            self.rpn = RegionProposalNetwork(
                cfg, in_channels, mode,
            )
            #---------------------------------#
            #   构建分类器网络
            #---------------------------------#
            self.head = VGG16RoIHead(cfg, classifier)
        elif cfg.arch.__contains__('res'):
            self.extractor, classifier = resnet(arch=cfg.arch,pretrained=cfg.pretrained)
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            in_channels = list(list(self.extractor.children())
                               [-1].children())[-1].conv1.in_channels
            self.rpn = RegionProposalNetwork(
                cfg, in_channels, mode,
            )
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.head = ResnetRoIHead(cfg, classifier)
            
    def forward(self, x, scale=1.):
        #---------------------------------#
        #   计算输入图片的大小
        #---------------------------------#
        img_size        = x.shape[2:]
        #---------------------------------#
        #   利用主干网络提取特征
        #---------------------------------#
        base_feature    = self.extractor.forward(x)

        #---------------------------------#
        #   获得建议框
        #---------------------------------#
        _, _, rois, roi_indices, _  = self.rpn.forward(base_feature, img_size, scale)
        #---------------------------------------#
        #   获得classifier的分类结果和回归结果
        #---------------------------------------#
        roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
