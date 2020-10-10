from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.base import Base as DatasetBase
from model import Model


class Evaluator(object):
    def __init__(self, dataset: DatasetBase, path_to_data_dir: str):
        super().__init__()
        self._dataset = dataset
        self._dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        self._path_to_data_dir = path_to_data_dir

    def evaluate_pck(self, model, device, stop_at=None):
        pck = []
        with torch.no_grad():
            for i, (image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch, vertices_batch) in enumerate(tqdm(self._dataloader)):
                image_batch = image_batch.to(device)
                scale_batch = scale_batch.to(device)
                bboxes_batch = bboxes_batch.to(device)
                vertices_batch = vertices_batch.to(device)
                assert image_batch.shape[0] == 1, 'do not use batch size more than 1 on evaluation'
                if labels_batch[0,0].item() == 0:
                    continue
                batch_size, _, image_height, image_width = image_batch.shape

                features = model.eval().features(image_batch)
                proposal_vertices, proposal_classes, proposal_transformers = model.eval().detection.forward(features, bboxes_batch)
                detection_bboxes, detection_classes, detection_probs, detection_vertices, detection_batch_indices = \
                    model.eval().detection.generate_detections(bboxes_batch, proposal_classes, proposal_transformers, proposal_vertices, image_width, image_height)

                detection_vertices = detection_vertices / scale_batch
                vertices_batch[0] = vertices_batch[0] / scale_batch
                bboxes_batch[0] = bboxes_batch[0] / scale_batch

                for vert,gt_vert,bbox in zip(detection_vertices, vertices_batch[0], bboxes_batch[0]):
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    max_dim = max(width.item(),height.item())
                    t_dist = [torch.norm(vert[:,j] - gt_vert[:,j], 2).item() for j in range(8)]
                    nb_correct = sum([d < 0.1*max_dim for d in t_dist])
                    pck.append(float(nb_correct/8))

                if stop_at is not None and i == stop_at:
                    break
        pck = float(sum(pck)/len(pck))
        return pck, 'pck'

    def evaluate(self, model: Model, device) -> Tuple[float, str]:
        all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs, all_detection_vertices = [], [], [], [], []

        with torch.no_grad():
            for i, (image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch, vertices_batch) in enumerate(tqdm(self._dataloader)):
                image_batch = image_batch.to(device)
                scale_batch = scale_batch.to(device)
                assert image_batch.shape[0] == 1, 'do not use batch size more than 1 on evaluation'

                detection_bboxes, detection_probs, detection_vertices, detection_batch_indices = \
                    model.eval().forward(image_batch)

                detection_bboxes = detection_bboxes / scale_batch
                detection_vertices = detection_vertices / scale_batch

                all_detection_bboxes.extend(detection_bboxes.tolist())
                all_detection_vertices.extend(detection_vertices.tolist())
                all_detection_probs.extend(detection_probs.tolist())
                all_image_ids.extend([image_id_batch[i] for i in detection_batch_indices])

        sort = sorted(zip(all_detection_probs,all_image_ids,all_detection_bboxes,all_detection_vertices), reverse=True)
        all_detection_probs = [x[0] for x in sort]
        all_image_ids = [x[1] for x in sort]
        all_detection_bboxes = [x[2] for x in sort]
        all_detection_vertices = [x[3] for x in sort]

        mean_ap, detail = self._dataset.evaluate(all_image_ids, all_detection_bboxes, all_detection_vertices, metric='ap')
        return mean_ap, detail