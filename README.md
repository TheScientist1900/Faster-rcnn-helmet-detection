## 基于Faster-rcnn的安全帽检测（使用Colab进行训练）
---
 
## 目录
1. [概述 Introduction](#概述)
2. [文件下载 Download](#文件下载)
3. [训练步骤 How2train](#训练步骤)
4. [评估步骤 How2eval](#评估步骤)
5. [参考资料 Reference](#Reference)


## 概述
使用Faster-RCNN进行安全帽检测，数据集比较简单，用来练手非常合适。

由于本人极度缺卡，因此选择在Colab进行训练。

也可以选择在本机进行训练，步骤在

本文对应的博客：https://blog.csdn.net/qq_43312130/article/details/123647987
## 文件下载
- 数据集地址：https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset

- 训练所需的voc_weights_resnet.pth或者voc_weights_vgg.pth以及主干的网络权重可以在百度云下载。  
voc_weights_resnet.pth是resnet为主干特征提取网络用到的；  
voc_weights_vgg.pth是vgg为主干特征提取网络用到的；   
链接: https://pan.baidu.com/s/1S6wG8sEXBeoSec95NZxmlQ      
提取码: 8mgp    
  
## 训练步骤
### a、数据预处理
1. 数据集的准备   
**下载安全帽数据集，并把其整理为`VOC`格式。**  

2. 准备类别文件

    首先在VOC2028文件夹下新建voc_classes.txt文件，包含数据集包含的类别的名字。顺序没有关系，但是名字一定要和xml文件里object里的name匹配，person不能在这里写成people。

3. 数据集的处理   

    修改`voc_annotation.py`里面的`annotation_mode=0`，运行voc_annotation.py生成根目录下的train.txt、val.txt、test.txt。   

    **ps**： 本文默认是在Colab上进行训练，如果选择在**本机**上训练，则修改`voc_annotation.py`里面`op=1`。这样生成的txt文件图像路径是本地的路径。

4. 数据可视化

    运行`utils\utils.py`文件。

### b、开始训练
1. 下载预训练模型
本文使用的backbone是vgg16，建议下载原博主用vgg16为backbone在VOC训练好的模型（https://pan.baidu.com/s/1S6wG8sEXBeoSec95NZxmlQ），放到model_data，加载这个模型后再训练安全帽数据集，效果会好很多。

2. 开始网络训练   
    新建一个Colaboratory，输入以下命令行即可。
    ```bash
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install yacs
    !python drive/MyDrive/faster-rcnn-pytorch-master/train.py
    ```
    **ps**：如果是在本机训练，则命令行输入：

    ```bash
    python drive/MyDrive/faster-rcnn-pytorch-master/train.py
    ```

3. 训练结果预测   
训练结果预测需要用到两个文件，分别是frcnn.py和predict.py。
`frcnn.py`中的`model_path`指向训练好的权值文件，这个修改为需要加载的模型的相对路径即可。   
 
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"    : r'model_best.pth',
    "classes_path"  : r'VOCdevkit\VOC2028\voc_classes.txt',
    #---------------------------------------------------------------------#
    #   网络的主干特征提取网络，resnet50或者vgg
    #---------------------------------------------------------------------#
    "backbone"      : "vgg",
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"    : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"       : 0.3,
    #---------------------------------------------------------------------#
    #   用于指定先验框的大小
    #---------------------------------------------------------------------#
    'anchors_size'    :[1, 4, 8, 16, 32, 128, 256, 512],
    'base_size' : 1,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"          : True,
}
```
  

## 评估步骤 
1. 本文使用VOC格式进行评估。  
2. 在frcnn.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，classes_path指向检测类别所对应的txt。**  
3. 运行`get_map.py`即可获得评估结果，评估结果会保存在map_out文件夹中。

## Reference
https://github.com/chenyuntc/simple-faster-rcnn-pytorch  
https://github.com/eriklindernoren/PyTorch-YOLOv3  
https://github.com/BobLiu20/YOLOv3_PyTorch
https://github.com/bubbliiiing/faster-rcnn-pytorch  
