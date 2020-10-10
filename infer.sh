#!/bin/bash
	
CUDA_VISIBLE_DEVICES=0 python infer.py -s=container -b=vgg16 -c=outputs/checkpoints-20201008185307-container-vgg16-ceff1921/container_model.pth data/container/images/MVI_3015.MP4 images/results/