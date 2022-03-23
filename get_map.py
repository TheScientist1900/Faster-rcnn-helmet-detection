import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from frcnn import FRCNN
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from utils.config import _C as cfg
if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，在门限值不同时，网络的Recall和Precision值是不同的。
    map计算结果中的Recall和Precision代表的是当预测时，门限置信度为0.5时，所对应的Recall和Precision值。

    此处获得的./map_out/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
    目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#

    #   MINOVERLAP用于指定想要获得的mAP0.x
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #-------------------------------------------------------#
    MINOVERLAP      = 0.5

    image_ids = open(cfg.test.map.data_file, 'r', encoding='UTF-8').readlines()
    image_ids = [img_id.strip() for img_id in image_ids]
    
    if not os.path.exists(cfg.test.map.map_out_dir):
        os.makedirs(cfg.test.map.map_out_dir)
    if not os.path.exists(os.path.join(cfg.test.map.map_out_dir, 'ground-truth')):
        os.makedirs(os.path.join(cfg.test.map.map_out_dir, 'ground-truth'))
    if not os.path.exists(os.path.join(cfg.test.map.map_out_dir, 'detection-results')):
        os.makedirs(os.path.join(cfg.test.map.map_out_dir, 'detection-results'))
    if not os.path.exists(os.path.join(cfg.test.map.map_out_dir, 'images-optional')):
        os.makedirs(os.path.join(cfg.test.map.map_out_dir, 'images-optional'))

    if cfg.test.map.mode == 0 or cfg.test.map.mode == 1:
        print("Load model.")
        frcnn = FRCNN(confidence = cfg.test.confidence, nms_iou = cfg.test.nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(cfg.data.root, "JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            if cfg.test.map.map_vis:
                image.save(os.path.join(cfg.test.map.map_out_dir, "images-optional/" + image_id + ".jpg"))
            frcnn.get_map_txt(image_id, image, cfg.data.class_names, cfg.test.map.map_out_dir)
        print("Get predict result done.")
        
    if cfg.test.map.mode == 0 or cfg.test.map.mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(cfg.test.map.map_out_dir, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(cfg.data.root, "Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in cfg.data.class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if cfg.test.map.mode == 0 or cfg.test.map.mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path = cfg.test.map.map_out_dir)
        print("Get map done.")

    if cfg.test.map.mode == 4:
        print("Get map.")
        get_coco_map(class_names = cfg.data.class_names, path = cfg.test.map.map_out_dir)
        print("Get map done.")
