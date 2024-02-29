# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import os
import tqdm
CUDA_VISIBLE_DEVICES=1
config = '/home/yzhang/AS-MLP-Object-Detection/work_dirs/qianqishiyan/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco.py'
checkpoint = '/home/yzhang/AS-MLP-Object-Detection/work_dirs/qianqishiyan/mask_rcnn_r50_fpn_1x_coco/epoch_12.pth'
img = '/home/yzhang/AS-MLP-Object-Detection/test_demo'
out_file='/home/yzhang/AS-MLP-Object-Detection/result_maskrcnn'
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', default='/home/yzhang/AS-MLP-Object-Detection/test_demo', help='Image file')
    parser.add_argument('config', default='/home/yzhang/AS-MLP-Object-Detection/work_dirs/qianqishiyan/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco.py', help='Config file')
    parser.add_argument('checkpoint', default='/home/yzhang/AS-MLP-Object-Detection/work_dirs/qianqishiyan/mask_rcnn_r50_fpn_1x_coco/epoch_12.pth', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--out_file', default='result_pointrend', help='Path to output file')
    
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    for filename in tqdm.tqdm(os.listdir(args.img)):
        img = os.path.join(args.img, filename)
        result = inference_detector(model, img)
        out_file = os.path.join(args.out_file, filename)
        model.show_result(
        img,
        result,
        score_thr=0.6,
        show=True,
        out_file=out_file,
        bbox_color='red',
        text_color='green',
        thickness=2,  
        font_size=10
        )


if __name__ == '__main__':
    args = parse_args()
    main(args)

