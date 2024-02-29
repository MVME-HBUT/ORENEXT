# ORENeXt
Offical codes for ["An Efficient MLP-based Point-guided Segmentation Network for Ore Images with Ambiguous Boundary"](https://arxiv.org/abs/2402.17370).

ORENeXt is based on mmdetection. The basic detector is [PointRend](https://github.com/open-mmlab/mmdetection/tree/main/configs/point_rend)
## Step 1: Installation
Clone this repo:

```
git clone https://github.com/pengyuting181/ORENeXt
cd ORENeXt
```

This repo is developed based on [mmdetection](https://github.com/open-mmlab/mmdetection).

Create a conda virtual environment and activate it:
```
conda create -n ORENeXt python=3.7 -y
conda activate ORENeXt
```

Dependencies
  - Linux or Windows
  - Python 3.7+, recommended 3.10
  - PyTorch 2.0 or higher, recommended 2.1
  - CUDA 11.7 or higher, recommended 12.1
  - MMCV 2.0 or higher, recommended 2.1

## Step 2: Prepare dataset
Prepare for ore dataset, you can get from [Orev1](https://drive.google.com/file/d/1eYkPHgDWULHind802P4tvy9l7lIQrpqk/view?pli=1.) 

## Step 3: Training and Evaluation

### Training
```
CUDA_VISIBLE_DEVICES=×× python ./tools/train.py ./configs/××.py   --cfg-options model.pretrained=./pretrained/××.pth
```
### Evaluation
```
CUDA_VISIBLE_DEVICES=×× python tools/test.py ./configs/××.py ./××.pth --eval bbox segm
```
##Visualize the results
```
python demo/image_demo.py test_demo/*.png configs/××.py work_dirs/××.pth --device cpu
```

##Cite
