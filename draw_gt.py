import argparse
import os
import random
import torch
import json
from tqdm import tqdm
import numpy as np
import cv2

from PIL import ImageDraw
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from bbox import BBox
from model import Model
from roi.pooler import Pooler
from config.eval_config import EvalConfig as Config

def draw(path_to_input_image, dataset_name):
    image = transforms.Image.open(path_to_input_image)
    dataset_class = DatasetBase.from_name(dataset_name)
    image_tensor, scale = dataset_class.preprocess(image, 600.0, 1000.0)
    #annotation_path = 'data/sunprimitive/annotations/val.json'
    annotation_path = 'data/container/vertices/MVI_3015.MP4.json'
    with open(annotation_path) as f:
        annotations = json.load(f)
    gt = annotations[path_to_input_image.split('/')[-1]]
    gt_vertices = [obj['vertices'] for obj in gt]

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'white']

    for vert in gt_vertices:

        quads = []
        quads.append(((int(vert[0][0]), int(vert[1][0])), (int(vert[0][1]), int(vert[1][1])), (int(vert[0][3]), int(vert[1][3])), (int(vert[0][2]), int(vert[1][2])), (int(vert[0][0]), int(vert[1][0]))))
        quads.append(((int(vert[0][0]), int(vert[1][0])), (int(vert[0][4]), int(vert[1][4])), (int(vert[0][5]), int(vert[1][5])), (int(vert[0][1]), int(vert[1][1])), (int(vert[0][0]), int(vert[1][0]))))
        quads.append(((int(vert[0][0]), int(vert[1][0])), (int(vert[0][4]), int(vert[1][4])), (int(vert[0][6]), int(vert[1][6])), (int(vert[0][2]), int(vert[1][2])), (int(vert[0][0]), int(vert[1][0]))))
        quads.append(((int(vert[0][1]), int(vert[1][1])), (int(vert[0][5]), int(vert[1][5])), (int(vert[0][7]), int(vert[1][7])), (int(vert[0][3]), int(vert[1][3])), (int(vert[0][1]), int(vert[1][1]))))
        quads.append(((int(vert[0][4]), int(vert[1][4])), (int(vert[0][5]), int(vert[1][5])), (int(vert[0][7]), int(vert[1][7])), (int(vert[0][6]), int(vert[1][6])), (int(vert[0][4]), int(vert[1][4]))))
        quads.append(((int(vert[0][2]), int(vert[1][2])), (int(vert[0][3]), int(vert[1][3])), (int(vert[0][7]), int(vert[1][7])), (int(vert[0][6]), int(vert[1][6])), (int(vert[0][2]), int(vert[1][2]))))

        for i in range(8):

            cv2.putText(image,str(i), (int(vert[0][i]), int(vert[1][i])), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)
        for i,quad in enumerate(quads):
            image = cv2.line(image, quad[0], quad[1], (255,255,0), 1) 
            image = cv2.line(image, quad[1], quad[2], (255,255,0), 1) 
            image = cv2.line(image, quad[2], quad[3], (255,255,0), 1) 
            image = cv2.line(image, quad[3], quad[0], (255,255,0), 1) 

    if len(gt_vertices) > 0:
        path_to_output_image = os.path.join('images/container_gt', path_to_input_image.split('/')[-1])
        cv2.imwrite(path_to_output_image, image)

if __name__=="__main__":

    folder = 'data/sunprimitive/val'
    folder = 'data/container/images/MVI_3015.MP4'
    files = os.listdir(folder)
    for file in tqdm(files):
        draw(os.path.join(folder, file), 'container')