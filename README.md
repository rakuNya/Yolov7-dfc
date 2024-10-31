# YOLOv7-dfc

This improved method for vehicle detection is based on the YOLOv7-tiny, it can be used in the same way as YOLOv7.

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)
<a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)

## Performance

| Model             |   FPS   | AP<sub>50</sub><sup> | GFlops | parameters |
|:------------------|:-------:|:--------------------:|:------:|------------|
| YOLOv7-tiny       | **131** |      **60.2%**       |  13.2  | 6.01M      |
| +Sim_DFC          | **91**  |      **66.2%**       |  15.3  | 7.56M      |
| ++Inner-shape IoU | **95**  |      **67.9%**       |  15.3  | 7.56M      |
| +++BiFPN          | **104** |      **70.0%**       |  14.8  | 7.46M      |

The result above is by UA-DETRAC dataset.

Our trained weight can be found in yolov7-dfc/improvedModel 

Dataset: https://openxlab.org.cn/datasets/OpenDataLab/Visdrone_DET, 

https://www.albany.edu/cnse/research/computer-vision-machine-learning-lab-tab-projects.

## Installation
The model has been tested and confirmed to run successfully in the following environment:

PyTorch 1.11.0,
Python 3.8,
CUDA 11.3.

Other library files required, run:
``` shell
pip install -r requirements.txt
```
For other virtual environments or hardware configurations with similar versions, there should generally be no compatibility issues. Please feel free to use it.
## Test
Specifically, you can view or apply our improved model by accessing the dfc-nn.yaml file under the path yolov7-dfc/cfg/deploy. This .yaml file can be called during training in train.py, where you may also select the appropriate dataset to test the model’s performance. Use UA-DETRAC dataset by default

``` shell
## in line 530 of train.py
parser.add_argument('--cfg', type=str, default='cfg/deploy/dfc-nn.yaml', help='model.yaml path')
parser.add_argument('--data', type=str, default='data/data.yaml', help='data.yaml path')
```




To use UA-DETRAC dataset (download) or other datasets, please change the of data.yaml in yolov7-dfc/data
```shell
# path
train: UA-DETRAC/images/train
val: UA-DETRAC/images/val
test: UA-DETRAC/images/test

# number of classes
nc: 4

# class names
names: ['car', 'bus', 'van','others']
```

For certain modules and network structures in the improved model, you can find the defined functions in common.py under the path yolov7-dfc/models, as well as in loss.py and general.py under yolov7-dfc/utils. By modifying these files, you can replace corresponding modules and adjust the structure to implement other customized operations.

Our improvements have been defined in these locations:
```shell
# in common.py
class Sim_DFC(nn.Module)
      () 
class BiFPN_Add2(nn.Module):
      ()
class BiFPN_Add3(nn.Module):
      ()
# in loss.py and general.py
class ComputeLoss:
      ()
def bbox_iou()
```

To train, run train.py, or run:
```shell
python train.py --workers 8 --batch-size 16 --data data/data.yaml --img 640 640 --weights '' 
```
The trained weights will be saved in runs/train/,where you need to find the best.pt file as the final weight.
Don't forget to set training epoch, it could lead to varies training results.

```shell
# in line 533
parser.add_argument('--epochs', type=int, default=100)
```

To test,  be cautious with the weight you choose. 
```shell
python test.py --data data/data.yaml --img 640 --batch 32 --weights improvedModel/weight.pt 
```

Or you can choose the path of weight in test.py:
```shell
# in line 293 of test.py
parser.add_argument('--weights', nargs='+', type=str, default='improvedModel/weight.pt', help='model.pt path(s)')
```
and run test.py. 

To inference, pay attention to the source file.
```shell
python detect.py --weights improvedModel/weight.pt  --source yourvideo.mp4
```

Or run detect.py with changes in code:
```shell
# in line 168, 169
parser.add_argument('--weights', nargs='+', type=str, default='improvedModel/weight.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='inference/image', help='source')
```

