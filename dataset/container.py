import json
import os
import pickle
import random
from typing import List, Tuple, Dict

import torch
import torch.utils.data.dataset
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np
from voc_eval import voc_ap

from bbox import BBox
from dataset.base import Base
from io import StringIO
import sys


class Container(Base):

    class Annotation(object):
        class Object(object):
            def __init__(self, bbox: BBox, label: int):
                super().__init__()
                self.bbox = bbox
                self.label = label

            def __repr__(self) -> str:
                return 'Object[label={:d}, bbox={!s}]'.format(
                    self.label, self.bbox)

        def __init__(self, filename: str, objects: List[Object]):
            super().__init__()
            self.filename = filename
            self.objects = objects

    CATEGORY_TO_LABEL_DICT = {'Non-Cuboid': 0, 'Cuboid': 1}

    LABEL_TO_CATEGORY_DICT = {v: k for k, v in CATEGORY_TO_LABEL_DICT.items()}

    def __init__(self, path_to_data_dir: str, mode: Base.Mode, image_min_side: float, image_max_side: float, val_folder='MVI_3015.MP4'):
        super().__init__(path_to_data_dir, mode, image_min_side, image_max_side)

        base_path = os.path.join(self._path_to_data_dir, 'container')
        self.images_base = os.path.join(base_path, 'images')
        self.annotations_base = os.path.join(base_path, 'vertices')

        if self._mode == Container.Mode.TRAIN:
            train_folders = os.listdir(self.images_base)
            train_folders.remove(val_folder)
            self.image_folders = train_folders
        elif self._mode == Container.Mode.EVAL:
            self.image_folders = [val_folder]
        else:
            raise ValueError('invalid mode')

        print(self.image_folders)

        self.annotations = {}
        self.img_paths = []
        for img_folder in self.image_folders:
            new_imgs = os.listdir(os.path.join(self.images_base, img_folder))
            self.img_paths.extend([os.path.join(img_folder, x) for x in new_imgs])
            with open(os.path.join(self.annotations_base, img_folder + '.json')) as f:
                self.annotations[img_folder] = json.load(f)
        self._image_ratios = []
        for img_path in self.img_paths:
            image = Image.open(os.path.join(self.images_base, img_path)).convert('RGB')  # for some grayscale images
            ratio = float(image.width / image.height)
            self._image_ratios.append(ratio)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        image_path = self.img_paths[index]
        image_name = image_path.split('/')[-1]
        image_folder = image_path.split('/')[0]
        annotation = self.annotations[image_folder][image_name]

        bboxes = [obj['bbox'] for obj in annotation]
        vertices = [obj['vertices'] for obj in annotation]
        labels = [0] if len(bboxes) == 0 else [1]*len(bboxes)
        if labels[-1] == 0:
            bboxes = torch.zeros((1,4), dtype=torch.float)
            vertices = torch.zeros((1,2,8), dtype=torch.float)
        else:
            bboxes = torch.tensor(bboxes, dtype=torch.float)
            vertices = torch.tensor(vertices, dtype=torch.float)

        labels = torch.tensor(labels, dtype=torch.long)

        image = Image.open(os.path.join(self.images_base, image_path)).convert('RGB')  # for some grayscale images

        # random flip on only training mode
        if self._mode == Container.Mode.TRAIN and random.random() > 0.5:
            image = ImageOps.mirror(image)
            if labels[-1] == 1:
                bboxes[:, [0, 2]] = image.width - bboxes[:, [2, 0]]  # index 0 and 2 represent `left` and `right` respectively
                vertices[:,0,:] = image.width - vertices[:,0,:] #Might need to check the correctness here
                for i,vert in enumerate(vertices):
                    if vert[0,2] < vert[0,6]:
                        vertex_copy = vert.clone()
                        vertices[i,:,0] = vertex_copy[:,2]
                        vertices[i,:,1] = vertex_copy[:,3]
                        vertices[i,:,2] = vertex_copy[:,0]
                        vertices[i,:,3] = vertex_copy[:,1]
                        vertices[i,:,4] = vertex_copy[:,6]
                        vertices[i,:,5] = vertex_copy[:,7]
                        vertices[i,:,6] = vertex_copy[:,4]
                        vertices[i,:,7] = vertex_copy[:,5]
                    else:
                        vertex_copy = vert.clone()
                        vertices[i,:,0] = vertex_copy[:,6]
                        vertices[i,:,1] = vertex_copy[:,7]

                        vertices[i,:,6] = vertex_copy[:,0]
                        vertices[i,:,7] = vertex_copy[:,1]


        image, scale = Container.preprocess(image, self._image_min_side, self._image_max_side)
        scale = torch.tensor(scale, dtype=torch.float)
        if bboxes.nelement() > 0:
            bboxes *= scale
            vertices *= scale

        return image_path, image, scale, bboxes, labels, vertices

    def evaluate(self, all_image_ids, all_detection_bboxes, all_detection_vertices, metric='apk'):
        if metric == 'ap':

            npos = 0
            for img_folder in self.annotations:
                for img_name in self.annotations[img_folder]:
                    npos += len(self.annotations[img_folder][img_name])
            
            tp = np.zeros(len(all_image_ids))
            fp = np.zeros(len(all_image_ids))
            y_scores = torch.ones(len(all_image_ids))
            for i,(img_id, bbox) in enumerate(zip(all_image_ids, all_detection_bboxes)):
                image_name = img_id.split('/')[-1]
                image_folder = img_id.split('/')[0]
                gt_bboxes = [obj['bbox'] for obj in self.annotations[image_folder][image_name]]
                if len(gt_bboxes) == 0:
                    fp[i] = 1
                    continue
                gt_bboxes = torch.Tensor(gt_bboxes)
                bbox = torch.Tensor(bbox).unsqueeze(0)
                ious = BBox.iou(bbox, gt_bboxes)
                max_ious, _ = ious.max(dim=2)
                if max_ious.item() > 0.5:
                    tp[i] = 1
                else:
                    fp[i] = 1

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

            score = voc_ap(rec, prec)

        return score, metric

    def pck(self, all_detection_vertices, gt_vertices, gt_bbox):

        pck = []

        for vert,gt_vert,bbox in zip(all_detection_vertices, gt_vertices, gt_bbox):
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            max_dim = max(width,height)
            vert_tensor = torch.Tensor(vert)
            gt_vert_tensor = torch.Tensor(gt_vert)
            t_dist = [torch.norm(vert_tensor[:,j] - gt_vert_tensor[:,j], 2) for j in range(8)]
            nb_correct = sum([d.item() < 0.1*max_dim for d in t_dist])
            pck.append(float(nb_correct/8))
        pck = float(sum(pck)/len(pck))
        return pck, 'pck'

    @property
    def image_ratios(self) -> List[float]:
        return self._image_ratios

    @staticmethod
    def num_classes() -> int:
        return 2