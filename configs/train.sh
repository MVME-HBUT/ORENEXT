#!/usr/bin/env bash

#./tools/dist_train2.sh ./configs/stones/mask_rcnn_r101_fpn_1x_coco.py 1

#./tools/dist_train2.sh ./configs/stones/mask_rcnn_x101_32x4d_fpn_1x_coco.py 1

#./tools/dist_train2.sh ./configs/stones/ms_rcnn_r50_fpn_1x_coco.py 1

#./tools/dist_train2.sh ./configs/stones/point_rend_r50_caffe_fpn_mstrain_1x_coco.py 1

#./tools/dist_train2.sh ./configs/stones/solo_r50_fpn_1x_coco.py 1

# ./tools/dist_train2.sh ./configs/stones/yolact_r50_1x8_coco.py 1

# ./tools/dist_train2.sh ./configs/stones/yolact_r101_1x8_coco.py 1

#./tools/dist_train2.sh ./configs/stones/queryinst_r50_fpn_1x_coco.py 1

#./tools/dist_train2.sh ./configs/stones/mask_rcnn_r50_fpn_groie_1x_coco.py 1

#./tools/dist_train2.sh ./configs/stones/mask_rcnn_r50_fpn_mocov2-pretrain_1x_coco.py 1

#./tools/dist_train2.sh ./configs/stones/mask_rcnn_r50_fpn_swav-pretrain_1x_coco.py 1

#./tools/dist_train.sh ./configs/asmlp_tfds/mask_rcnn_asmlp_small_emb64_patch4_shift5_drop-3_mstrain_512-700_adamw_20e_coco.py 1 --cfg-options model.pretrained=./pretrained/asmlp_small_patch4_shift5_224.pth

#./tools/dist_train.sh ./configs/asmlp_tfds/mask_rcnn_asmlp_base_patch4_shift5_drop-3_mstrain_512-700_adamw_20e_coco.py 1 --cfg-options model.pretrained=./pretrained/asmlp_base_patch4_shift5_224.pth

#./tools/dist_train.sh ./configs/asmlp_stone/mask_rcnn_asmlp_tiny_patch4_shift5_drop-3_mstrain_320-640_adamw_1x_coco.py 1 --no-validate --cfg-options model.pretrained=./pretrained/asmlp_tiny_patch4_shift5_224.pth

#  ./tools/dist_train.sh ./configs/asmlp_stone/mask_rcnn_asmlp_small_patch4_shift5_drop-3_mstrain_320-640_adamw_1x_coco.py 1 --no-validate --cfg-options model.pretrained=./pretrained/asmlp_small_patch4_shift5_224.pth

CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./configs/asmlp_stone/yoloact_asmlp_tiny_patch4_shift5_drop-3_fpn256_sgd_1x_coco.py 1 --cfg-options model.pretrained=./pretrained/asmlp_tiny_patch4_shift5_224.pth

#./tools/dist_train.sh ./configs/asmlp_tfds/mask_rcnn_asmlp_tiny_patch4_shift5_drop-3_mstrain_512-700_adamw_20e_coco.py 1 --cfg-options model.pretrained=./pretrained/asmlp_tiny_patch4_shift5_224.pth
