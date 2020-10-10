#!/usr/bin/env bash 

# CUDA_VISIBLE_DEVICES=1 python train.py -s=sunprimitive -b=vgg16 --num_steps_to_snapshot=10000 --num_steps_to_display=100 --batch_size=2
CUDA_VISIBLE_DEVICES=1 python train.py -s=container -b=vgg16 -r=outputs/checkpoints-20201007103845-sunprimitive-vgg16-6b0a977f/model.pth --num_steps_to_finish=70000 --num_steps_to_snapshot=10000 --num_steps_to_display=100 --batch_size=2