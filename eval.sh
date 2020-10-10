#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0 python eval.py -s=sunprimitive -b=vgg16 --rpn_post_nms_top_n=1000 outputs/checkpoints-20201004122720-sunprimitive-vgg16-9e0a2b22/model.pth #batch 2, with I: loss=0.0086, pck=0.3987
# CUDA_VISIBLE_DEVICES=1 python eval.py -s=sunprimitive -b=vgg16 outputs/checkpoints-20201006144909-sunprimitive-vgg16-3195e308/model.pth #batch 2, with I: loss=0.009, pck=0.34
# CUDA_VISIBLE_DEVICES=0 python eval.py -s=sunprimitive -b=vgg16 --rpn_post_nms_top_n=1000 outputs/checkpoints-20201007103845-sunprimitive-vgg16-6b0a977f/model.pth #batch 2, with I: loss=0.0087, pck=0.4039, mAP=0.7509
CUDA_VISIBLE_DEVICES=0 python eval.py -s=container -b=vgg16 --rpn_post_nms_top_n=1000 outputs/checkpoints-20201008185307-container-vgg16-ceff1921/container_model.pth #vertex loss=0.0048, loss=0.077 pck=0.62, mAP=1