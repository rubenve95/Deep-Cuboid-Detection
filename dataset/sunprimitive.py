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


class SunPrimitive(Base):

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

    def __init__(self, path_to_data_dir: str, mode: Base.Mode, image_min_side: float, image_max_side: float):
        super().__init__(path_to_data_dir, mode, image_min_side, image_max_side)

        path_to_sunprimitive_dir = os.path.join(self._path_to_data_dir, 'sunprimitive')
        path_to_annotations_dir = os.path.join(path_to_sunprimitive_dir, 'annotations')
        path_to_caches_dir = os.path.join('caches', 'sunprimitive', f'{self._mode.value}')
        path_to_image_ids_pickle = os.path.join(path_to_caches_dir, 'image-ids.pkl')
        path_to_image_id_dict_pickle = os.path.join(path_to_caches_dir, 'image-id-dict.pkl')
        #path_to_image_ratios_pickle = os.path.join(path_to_caches_dir, 'image-ratios.pkl')

        if self._mode == SunPrimitive.Mode.TRAIN:
            self.path_to_jpeg_images_dir = os.path.join(path_to_sunprimitive_dir, 'train')
            path_to_annotation = os.path.join(path_to_annotations_dir, 'train.json')
        elif self._mode == SunPrimitive.Mode.EVAL:
            self.path_to_jpeg_images_dir = os.path.join(path_to_sunprimitive_dir, 'val')
            path_to_annotation = os.path.join(path_to_annotations_dir, 'val.json')
        else:
            raise ValueError('invalid mode')

        #coco_dataset = CocoDetection(root=path_to_jpeg_images_dir, annFile=path_to_annotation)

        if os.path.exists(path_to_image_ids_pickle) and os.path.exists(path_to_image_id_dict_pickle):
            print('loading cache files...')

            with open(path_to_image_ids_pickle, 'rb') as f:
                self.img_names = pickle.load(f)

            with open(path_to_image_id_dict_pickle, 'rb') as f:
                self.annotations = pickle.load(f)

            # with open(path_to_image_ratios_pickle, 'rb') as f:
            #     self._image_ratios = pickle.load(f)
        else:
            #print('generating cache files...')

            os.makedirs(path_to_caches_dir, exist_ok=True)

            self.img_names = os.listdir(self.path_to_jpeg_images_dir)
            with open(path_to_annotation) as f:
                self.annotations = json.load(f)

            self._image_ratios = []
            for img_name in self.img_names:
                image = Image.open(os.path.join(self.path_to_jpeg_images_dir, img_name)).convert('RGB')  # for some grayscale images
                ratio = float(image.width / image.height)
                self._image_ratios.append(ratio)

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        image_name = self.img_names[index]
        annotation = self.annotations[image_name]

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

        image = Image.open(os.path.join(self.path_to_jpeg_images_dir, image_name)).convert('RGB')  # for some grayscale images

        # random flip on only training mode
        if self._mode == SunPrimitive.Mode.TRAIN and random.random() > 0.5:
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


        image, scale = SunPrimitive.preprocess(image, self._image_min_side, self._image_max_side)
        scale = torch.tensor(scale, dtype=torch.float)
        if bboxes.nelement() > 0:
            bboxes *= scale
            vertices *= scale

        return image_name, image, scale, bboxes, labels, vertices

    def evaluate(self, all_image_ids, all_detection_bboxes, all_detection_vertices, metric='apk'):

        if metric == 'apk':
        #apk = []
            y_true = np.zeros(len(all_image_ids)*8)

            for i,(img_id, vert, bbox) in enumerate(zip(all_image_ids, all_detection_vertices, all_detection_bboxes)):
                best_nb_correct = -1
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                max_dim = max(width,height)
                for ann in self.annotations[img_id]:
                    gt_vert = ann['vertices']

                    vert_tensor = torch.Tensor(vert)
                    gt_vert_tensor = torch.Tensor(gt_vert)
                    t_dist = [torch.norm(vert_tensor[:,j] - gt_vert_tensor[:,j], 2) for j in range(8)]

                    correct = [d.item() < 0.1*max_dim for d in t_dist]
                    nb_correct = sum(correct)
                    if nb_correct > best_nb_correct:
                        best_nb_correct = nb_correct#max(nb_correct, best_nb_correct)
                        y_true[i:i+8] = np.asarray(correct)

            y_scores = np.ones(len(all_image_ids)*8)
            score = np.sum(y_true)/np.sum(y_scores)

        elif metric == 'ap':

            npos = 0
            for img_name in self.annotations:
                npos += len(self.annotations[img_name])
            
            tp = np.zeros(len(all_image_ids))
            fp = np.zeros(len(all_image_ids))
            y_scores = torch.ones(len(all_image_ids))
            for i,(img_id, bbox) in enumerate(zip(all_image_ids, all_detection_bboxes)):
                gt_bboxes = [obj['bbox'] for obj in self.annotations[img_id]]
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