import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from nets.frcnn_training import FasterRCNNTrainer, weights_init
from utils.callbacks import LossHistory
from utils.config import update_config
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import create_logger, get_classes
from utils.utils_fit import fit_one_epoch
import argparse
from utils.config import _C as cfg
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser(description='Faster RCNN')
    parser.add_argument('--data_name', default='VOC2028',
                        help='data dir')

    parser.add_argument('--arch', default='vgg16')
    parser.add_argument(
        '--checkpoint', default='model_data/voc_weights_vgg.pth', help='pretrained model')
    parser.add_argument('--pretrained', default=True, )
    return parser.parse_args()


def main(cfg):
    arg = parse_arg()
    cfg = update_config(cfg, arg)

    final_output_dir, logger = create_logger(cfg)
    logger.info(cfg)
    # cudnn related setting
    cudnn.benchmark = cfg.cudnn.benchmark  # True
    torch.backends.cudnn.deterministic = cfg.cudnn.deterministic  # False
    torch.backends.cudnn.enabled = cfg.cudnn.enabled  # True

    train_dataset = FRCNNDataset(cfg, mode='train', )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
        collate_fn=frcnn_dataset_collate)
    train_step_num = train_dataset.__len__() // cfg.train.batch_size

    val_dataset = FRCNNDataset(cfg, mode='val', )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
        collate_fn=frcnn_dataset_collate)
    val_step_num = val_dataset.__len__() // cfg.train.batch_size

    model = FasterRCNN(cfg=cfg, mode='train')
    if cfg.cuda:
        # model = torch.nn.DataParallel(model)
        model = model.cuda()

    begin_epoch = 0
    lowest_loss = 100
    freeze = 'freeze training' if cfg.train.freeze else 'unfreeze training'
    resume = False

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.freeze_lr, weight_decay=cfg.train.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.96)
    model.freeze_bn()
    if cfg.train.freeze:
        for param in model.extractor.parameters():
            param.requires_grad = False

    if cfg.checkpoint != '':
        # cfg.checkpoint = os.path.join(cfg.root, cfg.checkpoint)
        logger.info('=> load weights from {}'.format(
            os.path.join(cfg.root, cfg.checkpoint)))
        cpk = torch.load(os.path.join(cfg.root, cfg.checkpoint))

        if 'state_dict' in cpk.keys():
            logger.info('=> load params and continue training')
            model_pre_dict = cpk['state_dict']
            model.load_state_dict(model_pre_dict)
            begin_epoch = cpk['epoch'] + 1
            lr_scheduler.load_state_dict(cpk['lr_scheduler'])
            optimizer.load_state_dict(cpk['optimizer'])
            lowest_loss = cpk['loss']

            resume = True
            if cfg.train.freeze and begin_epoch >= cfg.cfg.train.freeze_epoch:
                freeze = 'unfreeze training'
        else:
            logger.info('=> load VOC weights and start training')
            model_dict = model.state_dict()
            model_pre_dict = cpk
            model_pre_dict = {k: v for k, v in model_pre_dict.items() 
                                if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(model_pre_dict)
            model.load_state_dict(model_dict)

    logger.info('=> '+freeze)

    train_util = FasterRCNNTrainer(cfg, model, optimizer)
    loss_history = LossHistory(final_output_dir)

    for epoch in range(begin_epoch, cfg.train.epoch):
        if cfg.train.freeze and epoch >= cfg.train.freeze_epoch and \
                (freeze == 'freeze training' or resume):
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.train.unfreeze_lr, weight_decay=cfg.train.wd)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.96)
            for param in model.extractor.parameters():
                param.requires_grad = True
            train_util = FasterRCNNTrainer(cfg, model, optimizer)

            train_loader.batch_size = int(cfg.train.batch_size/4)
            train_step_num = train_dataset.__len__() // train_loader.batch_size

            val_loader.batch_size = int(cfg.train.batch_size/4)
            val_step_num = val_dataset.__len__() // val_loader.batch_size
            train_type = 'unfreeze training'
            
            logger.info(train_type)

        # trainer.train_epoch(train_loader, train_step_num, epoch, cfg.train.epoch, final_output_dir, lr_scheduler)
        train_loss, val_loss = fit_one_epoch(model, train_util, loss_history, optimizer, epoch,
                                             train_step_num, val_step_num, train_loader, val_loader, 
                                             cfg.train.epoch, cfg.cuda, logger,)

        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'loss': lowest_loss
        }, os.path.join(final_output_dir, 'checkpoint.pth'), _use_new_zipfile_serialization=False)
        # )
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'loss': lowest_loss
            }, os.path.join(final_output_dir, 'model_best.pth'), _use_new_zipfile_serialization=False)
            # )
        lr_scheduler.step()


if __name__ == "__main__":
    main(cfg)
