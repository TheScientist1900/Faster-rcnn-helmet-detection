import logging
import pathlib
import time
import numpy as np
from PIL import Image
import cv2
import os

from tqdm import tqdm
#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#


def resize_image(image, size):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image


def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width


def draw_annotation(img_path, annot_path, save_path, classes_path, batch=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    class_names, num_classes = get_classes(classes_path)
    if batch:
        print('start draw ground truth in batch way')
        f = open(annot_path, 'r', encoding='UTF-8')
        annot = f.readlines()
        num = 0
        
        for line in tqdm(annot):
            line = line.split()

            img = cv2.imread(line[0])
            img_name = os.path.basename(line[0])
            img_copy = img.copy()
            for box in line[1:]:
                box = box.split(',')
                cv2.rectangle(img_copy, (int(box[0]), int(box[1])), (int(
                    box[2]), int(box[3])), color=(0, 255, 0), thickness=3)
                label = class_names[int(box[-1])]
                cv2.putText(img_copy, label, (int(box[0]), int(box[1])), font,
                            1.2, (0, 0, 255), 2)
            if not os.path.exists(save_path):
                os.mkdir(os.path.join(save_path))
            cv2.imwrite(os.path.join(save_path, img_name), img_copy)
            num += 1
        print('draw {} ground truth in batch way done!'.format(num))
    else:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        # img = img[:,:,::-1]

        f = open(annot_path, 'r')
        annot = f.readlines()

        a = img.copy()
        for line in annot:
            line = line.split()
            annot_name = os.path.basename(line[0])
            if annot_name[:-4] == img_name[:-4]:
                for box in line[1:]:
                    box = box.split(',')
                    cv2.rectangle(a, (int(box[0]), int(box[1])), (int(
                        box[2]), int(box[3])), color=(0, 0, 255))
        cv2.imshow('1', a)
        cv2.waitKey(0)

def create_logger(cfg):
    
    final_output_dir = pathlib.Path(cfg.root) / pathlib.Path('exp') /  cfg.data_name / cfg.arch
    print("=> creating dir {}".format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)
    final_log_file = final_output_dir / log_file

    head = "%(asctime)-15s\n%(message)s"
    logging.basicConfig(filename=final_log_file, format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)
    logger.info("=> creating log file {}".format(final_log_file))

    return final_output_dir, logger

if __name__ == '__main__':
    # 把这个函数跑三次，路径也要相应的改
    # 第一次
    draw_annotation(img_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\JPEGImages', 
                    annot_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\train.txt', 
                    save_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\Trainval-Ground-truth', 
                    classes_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\voc_classes.txt', 
                    batch=True)

    draw_annotation(img_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\JPEGImages', 
                    annot_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\val.txt', 
                    save_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\Trainval-Ground-truth', 
                    classes_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\voc_classes.txt', 
                    batch=True)
    draw_annotation(img_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\JPEGImages', 
                    annot_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\test.txt', 
                    save_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\Test-Ground-truth', 
                    classes_path=r'your-path\faster-rcnn-pytorch-master\VOCdevkit\VOC2028\voc_classes.txt', 
                    batch=True)
    
