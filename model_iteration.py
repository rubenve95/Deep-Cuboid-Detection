import os
from typing import Union, Tuple, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from backbone.base import Base as BackboneBase
from bbox import BBox
from extension.functional import beta_smooth_l1_loss
from roi.pooler import Pooler
from rpn.region_proposal_network import RegionProposalNetwork
from support.layer.nms import nms


class Model(nn.Module):

    def __init__(self, backbone: BackboneBase, num_classes: int, pooler_mode: Pooler.Mode,
                 anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
                 rpn_pre_nms_top_n: int, rpn_post_nms_top_n: int,
                 anchor_smooth_l1_loss_beta: Optional[float] = None, proposal_smooth_l1_loss_beta: Optional[float] = None, iteration=True):
        super().__init__()

        self.features, hidden, num_features_out, num_hidden_out = backbone.features()
        self._bn_modules = nn.ModuleList([it for it in self.features.modules() if isinstance(it, nn.BatchNorm2d)] +
                                         [it for it in hidden.modules() if isinstance(it, nn.BatchNorm2d)])

        # NOTE: It's crucial to freeze batch normalization modules for few batches training, which can be done by following processes
        #       (1) Change mode to `eval`
        #       (2) Disable gradient (we move this process into `forward`)
        for bn_module in self._bn_modules:
            for parameter in bn_module.parameters():
                parameter.requires_grad = False

        self.rpn = RegionProposalNetwork(num_features_out, anchor_ratios, anchor_sizes, rpn_pre_nms_top_n, rpn_post_nms_top_n, anchor_smooth_l1_loss_beta)
        self.detection = Model.Detection(pooler_mode, hidden, num_hidden_out, num_classes, proposal_smooth_l1_loss_beta, iteration=iteration)
        self.iteration = iteration

    def forward(self, image_batch: Tensor,
                gt_bboxes_batch: Tensor = None, gt_classes_batch: Tensor = None, gt_vertices_batch: Tensor = None):
        # disable gradient for each forwarding process just in case model was switched to `train` mode at any time
        for bn_module in self._bn_modules:
            bn_module.eval()

        features = self.features(image_batch)

        batch_size, _, image_height, image_width = image_batch.shape
        _, _, features_height, features_width = features.shape

        anchor_bboxes = self.rpn.generate_anchors(image_width, image_height, num_x_anchors=features_width, num_y_anchors=features_height).to(features).repeat(batch_size, 1, 1)

        if self.training:
            anchor_objectnesses, anchor_transformers, anchor_objectness_losses, anchor_transformer_losses = self.rpn.forward(features, anchor_bboxes, gt_bboxes_batch, image_width, image_height)

            proposal_bboxes = self.rpn.generate_proposals(anchor_bboxes, anchor_objectnesses, anchor_transformers, image_width, image_height).detach()  # it's necessary to detach `proposal_bboxes` here

            proposal_vertices, proposal_classes, proposal_transformers, proposal_class_losses, proposal_transformer_losses, vertex_losses = \
                self.detection.forward(features, proposal_bboxes, gt_classes_batch, gt_bboxes_batch, gt_vertices_batch, image_width=image_width, image_height=image_height)

            return anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses, vertex_losses
        else:
            anchor_objectnesses, anchor_transformers = self.rpn.forward(features)
            proposal_bboxes = self.rpn.generate_proposals(anchor_bboxes, anchor_objectnesses, anchor_transformers, image_width, image_height)
            proposal_vertices, proposal_classes, proposal_transformers, proposal_transformers2 = self.detection.forward(features, proposal_bboxes, image_width=image_width, image_height=image_height)
            detection_bboxes, detection_classes, detection_probs, detection_vertices, detection_batch_indices = \
                self.detection.generate_detections(proposal_bboxes, proposal_classes, proposal_transformers, proposal_vertices, image_width, image_height, proposal_transformers2)
            return detection_bboxes, detection_probs, detection_vertices, detection_batch_indices

    def save(self, path_to_checkpoints_dir: str, step: int, optimizer: Optimizer, scheduler: _LRScheduler) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir, f'model.pth')
        checkpoint = {
            'state_dict': self.state_dict(),
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str, optimizer: Optimizer = None, scheduler: _LRScheduler = None) -> 'Model':
        checkpoint = torch.load(path_to_checkpoint)
        self.load_state_dict(checkpoint['state_dict'])
        step = checkpoint['step']
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return step

    class Detection(nn.Module):

        def __init__(self, pooler_mode: Pooler.Mode, hidden: nn.Module, num_hidden_out: int, num_classes: int, proposal_smooth_l1_loss_beta: float, iteration=True):
            super().__init__()
            self._pooler_mode = pooler_mode
            self.hidden = hidden
            self.num_classes = num_classes
            self._proposal_class = nn.Linear(num_hidden_out, 1)
            self._proposal_transformer = nn.Linear(num_hidden_out, 4)
            self._proposal_vertices = nn.Linear(num_hidden_out, 16)
            self._proposal_smooth_l1_loss_beta = proposal_smooth_l1_loss_beta
            self._transformer_normalize_mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float)
            self._transformer_normalize_std = torch.tensor([.1, .1, .2, .2], dtype=torch.float)
            self.dropout = nn.Dropout(0.5)
            self.iteration=iteration

        def forward(self, features: Tensor, proposal_bboxes: Tensor, gt_classes_batch: Optional[Tensor] = None, gt_bboxes_batch: Optional[Tensor] = None, 
                gt_vertices_batch: Optional[Tensor] = None, image_width = None, image_height = None) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
            batch_size = features.shape[0]

            if not self.training:
                proposal_batch_indices = torch.arange(end=batch_size, dtype=torch.long, device=proposal_bboxes.device).view(-1, 1).repeat(1, proposal_bboxes.shape[1])
                pool = Pooler.apply(features, proposal_bboxes.view(-1, 4), proposal_batch_indices.view(-1), mode=self._pooler_mode)
                pool = pool.view(pool.shape[0], -1)
                hidden = self.hidden[0](pool)
                hidden = self.hidden[1](hidden)
                hidden = self.hidden[3](hidden)
                hidden = self.hidden[4](hidden)
                #hidden = F.adaptive_max_pool2d(input=hidden, output_size=1)
                #hidden = hidden.view(hidden.shape[0], -1)
                proposal_transformers = self._proposal_transformer(hidden)

                if self.iteration:
                    detection_bboxes = self.create_bboxes(proposal_bboxes, proposal_transformers.unsqueeze(0), image_width, image_height, 1)
                    detection_bboxes = detection_bboxes.view(-1,4)
                    pool = Pooler.apply(features, detection_bboxes.view(-1,4), proposal_batch_indices.view(-1), mode=self._pooler_mode)
                    pool = pool.view(pool.shape[0], -1)

                    hidden = self.hidden[0](pool)
                    hidden = self.hidden[1](hidden)
                    hidden = self.hidden[3](hidden)
                    hidden = self.hidden[4](hidden)
                    proposal_transformers2 = self._proposal_transformer(hidden)
                    proposal_transformers2 = proposal_transformers2.view(batch_size, -1, proposal_transformers.shape[-1])
                else:
                    proposal_transformers2 = None

                proposal_classes = self._proposal_class(hidden)
                proposal_vertices = self._proposal_vertices(hidden)

                proposal_classes = proposal_classes.view(batch_size, -1, proposal_classes.shape[-1])
                proposal_transformers = proposal_transformers.view(batch_size, -1, proposal_transformers.shape[-1])
                proposal_vertices = proposal_vertices.view(batch_size, -1, proposal_vertices.shape[-1])
                return proposal_vertices, proposal_classes, proposal_transformers, proposal_transformers2
            else:
                labels = torch.full((batch_size, proposal_bboxes.shape[1]), -1, dtype=torch.long, device=proposal_bboxes.device)
                # print(proposal_bboxes.size(), gt_bboxes_batch.size())
                ious = BBox.iou(proposal_bboxes, gt_bboxes_batch)
                #print(proposal_bboxes.size(), gt_bboxes_batch.size(), ious.size())
                proposal_max_ious, proposal_assignments = ious.max(dim=2)
                labels[proposal_max_ious < 0.5] = 0
                fg_masks = proposal_max_ious >= 0.5
                if len(fg_masks.nonzero()) > 0:
                    labels[fg_masks] = gt_classes_batch[fg_masks.nonzero()[:, 0], proposal_assignments[fg_masks]]

                # select 128 x `batch_size` samples
                fg_indices = (labels > 0).nonzero()
                bg_indices = (labels == 0).nonzero()
                fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 32 * batch_size)]]
                bg_indices = bg_indices[torch.randperm(len(bg_indices))[:128 * batch_size - len(fg_indices)]]
                selected_indices = torch.cat([fg_indices, bg_indices], dim=0)
                selected_indices = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)

                proposal_bboxes = proposal_bboxes[selected_indices]
                gt_bboxes = gt_bboxes_batch[selected_indices[0], proposal_assignments[selected_indices]]
                gt_vertices = gt_vertices_batch[selected_indices[0], proposal_assignments[selected_indices]]
                gt_proposal_classes = labels[selected_indices]
                gt_proposal_transformers = BBox.calc_transformer(proposal_bboxes, gt_bboxes)
                batch_indices = selected_indices[0]

                #print('before', gt_proposal_classes)
                #print(gt_proposal_classes.size())

                pool = Pooler.apply(features, proposal_bboxes, proposal_batch_indices=batch_indices, mode=self._pooler_mode)

                #vgg16
                hidden = self.hidden(pool.view(pool.shape[0], -1))

                #resnet101
                # hidden = self.hidden(pool)
                # hidden = F.adaptive_max_pool2d(input=hidden, output_size=1)
                # hidden = hidden.view(hidden.shape[0], -1)


                if self.iteration:
                    proposal_transformers_first_iter = self._proposal_transformer(hidden)
                    detection_bboxes = self.create_bboxes(proposal_bboxes.unsqueeze(0), proposal_transformers_first_iter.unsqueeze(0), image_width, image_height, 1)
                    detection_bboxes = detection_bboxes.view(-1,4)
                    pool = Pooler.apply(features, detection_bboxes, proposal_batch_indices=batch_indices, mode=self._pooler_mode)
                    hidden = self.hidden(pool.view(pool.shape[0], -1))

                    bboxes_centers = torch.stack(( (detection_bboxes[:,0] + detection_bboxes[:,2])/2,(detection_bboxes[:,1] + detection_bboxes[:,3])/2), dim=1)
                    width = detection_bboxes[:,2] - detection_bboxes[:,0]
                    height = detection_bboxes[:,3] - detection_bboxes[:,1]

                    gt_proposal_transformers = BBox.calc_transformer(detection_bboxes, gt_bboxes).detach()
                    for batch_index in range(batch_size):
                        selected_batch_indices = (batch_indices == batch_index).nonzero().view(-1)
                        ious = BBox.iou(detection_bboxes[selected_batch_indices].unsqueeze(0), gt_bboxes_batch[batch_index].unsqueeze(0)).detach()
                        #print('iter', detection_bboxes.size(), gt_bboxes.size(), ious.size())
                        max_ious, _ = ious.max(dim=2)
                    # print(gt_proposal_classes.size(), max_ious.size())
                        gt_proposal_classes[selected_batch_indices][max_ious[0] < 0.5] = 0
                        gt_proposal_classes[selected_batch_indices][max_ious[0] >= 0.5] = 1
                    #print('after', gt_proposal_classes)
                    #print(gt_proposal_classes.size())

                    # #if fg_indices.nelement() > 0:
                    # infinites = torch.isinf(gt_proposal_transformers)
                    # if gt_bboxes[gt_bboxes > 0].nelement() > 0 and infinites[infinites == 1].nelement() > 0:
                    #     #print(infinites)
                    #     #print(gt_proposal_transformers)
                    #     # print(infinites.size())
                    #     indices = torch.max(infinites,1)[0]
                    #     #print(indices)
                    #     indices = indices.nonzero().view(-1)

                        #print(indices)
                        #print('gt_proposal_transformers', gt_proposal_transformers[indices])
                        #print('detection_bboxes', detection_bboxes[indices])
                        #print('gt_bboxes', gt_bboxes[indices])
                        #print('ious', ious[0,index], ious.size())
                        #print(ious.size())
                        #print('max_ious', max_ious[0,indices], max_ious.size())
                        #print('gt_proposal_classes', gt_proposal_classes[indices], gt_proposal_classes.size())
                    #     #yo = BBox.calc_transformer(detection_bboxes[index].unsqueeze(0), gt_bboxes[index].unsqueeze(0), print_it=True).detach()


                else:
                    bboxes_centers = torch.stack(( (proposal_bboxes[:,0] + proposal_bboxes[:,2])/2,(proposal_bboxes[:,1] + proposal_bboxes[:,3])/2), dim=1)
                    width = proposal_bboxes[:,2] - proposal_bboxes[:,0]
                    height = proposal_bboxes[:,3] - proposal_bboxes[:,1]

                gt_vertices_norm = torch.empty(gt_vertices.size(), dtype=torch.float, device=gt_vertices.device)

                for i in range(gt_vertices_norm.size()[-1]):
                    gt_vertices_norm[:,:,i] = torch.stack(((gt_vertices[:,0,i] - bboxes_centers[:,0])/width,(gt_vertices[:,1,i] - bboxes_centers[:,1])/height), dim=1)
                gt_vertices_norm = gt_vertices_norm.detach()

                proposal_classes = self._proposal_class(hidden)
                proposal_vertices = self._proposal_vertices(hidden)
                proposal_transformers = self._proposal_transformer(hidden)

                proposal_class_losses, proposal_transformer_losses, vertex_losses = self.loss(proposal_vertices, proposal_classes, 
                                                                                proposal_transformers, gt_proposal_classes, 
                                                                                gt_proposal_transformers, gt_vertices_norm, batch_size, batch_indices)

                return proposal_vertices, proposal_classes, proposal_transformers, proposal_class_losses, proposal_transformer_losses, vertex_losses

        def loss(self, proposal_vertices: Tensor, proposal_classes: Tensor, proposal_transformers: Tensor,
                 gt_proposal_classes: Tensor, gt_proposal_transformers: Tensor, gt_vertices: Tensor,
                 batch_size, batch_indices) -> Tuple[Tensor, Tensor]:
            #proposal_transformers = proposal_transformers.view(-1, 2, 4)[torch.arange(end=len(proposal_transformers), dtype=torch.long), gt_proposal_classes]
            transformer_normalize_mean = self._transformer_normalize_mean.to(device=gt_proposal_transformers.device)
            transformer_normalize_std = self._transformer_normalize_std.to(device=gt_proposal_transformers.device)
            gt_proposal_transformers = (gt_proposal_transformers - transformer_normalize_mean) / transformer_normalize_std  # scale up target to make regressor easier to learn

            cross_entropies = torch.empty(batch_size, dtype=torch.float, device=proposal_classes.device)
            smooth_l1_losses = torch.empty(batch_size, dtype=torch.float, device=proposal_transformers.device)
            vertex_losses = torch.empty(batch_size, dtype=torch.float, device=proposal_vertices.device)

            bceloss = nn.BCELoss()
            sigmoid = nn.Sigmoid()

            for batch_index in range(batch_size):
                selected_indices = (batch_indices == batch_index).nonzero().view(-1)

                # print(proposal_classes)
                # print(sigmoid(proposal_classes[selected_indices]).squeeze(1))
                # print(gt_proposal_classes[selected_indices].float())
                cross_entropy = bceloss(sigmoid(proposal_classes[selected_indices]).squeeze(1), gt_proposal_classes[selected_indices].float())

                fg_indices = gt_proposal_classes[selected_indices].nonzero().view(-1)

                #print(proposal_transformers[selected_indices][fg_indices])


                corner_l1_loss = beta_smooth_l1_loss(input=proposal_transformers[selected_indices][fg_indices],
                                                     target=gt_proposal_transformers[selected_indices][fg_indices],
                                                     beta=self._proposal_smooth_l1_loss_beta)

                vertex_loss = beta_smooth_l1_loss(input=proposal_vertices[selected_indices][fg_indices],
                                                     target=gt_vertices[selected_indices][fg_indices].view(-1,16),
                                                     beta=self._proposal_smooth_l1_loss_beta)

                cross_entropies[batch_index] = cross_entropy
                smooth_l1_losses[batch_index] = corner_l1_loss
                vertex_losses[batch_index] = vertex_loss

            #print('losses', cross_entropies, smooth_l1_losses, vertex_losses)

            return cross_entropies, smooth_l1_losses, vertex_losses

        def create_bboxes(self, proposal_bboxes, proposal_transformers, image_width, image_height, batch_size):
            proposal_transformers = proposal_transformers.view(batch_size, -1, 4)
            proposal_bboxes = proposal_bboxes.unsqueeze(dim=2).repeat(1, 1, 1, 1)
            transformer_normalize_std = self._transformer_normalize_std.to(device=proposal_transformers.device)
            transformer_normalize_mean = self._transformer_normalize_mean.to(device=proposal_transformers.device)
            proposal_transformers = proposal_transformers * transformer_normalize_std + transformer_normalize_mean

            detection_bboxes = BBox.apply_transformer(proposal_bboxes, proposal_transformers.unsqueeze(2))
            detection_bboxes = BBox.clip(detection_bboxes, left=0, top=0, right=image_width, bottom=image_height)

            return detection_bboxes

        def generate_detections(self, proposal_bboxes: Tensor, proposal_classes: Tensor, proposal_transformers: Tensor, proposal_vertices: Tensor, image_width: int, image_height: int, proposal_transformers2 = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            batch_size = proposal_bboxes.shape[0]

            detection_bboxes = self.create_bboxes(proposal_bboxes, proposal_transformers, image_width, image_height, batch_size)
            if self.iteration:
                detection_bboxes2 = self.create_bboxes(detection_bboxes.squeeze(2), proposal_transformers2, image_width, image_height, batch_size)

            detection_probs = torch.sigmoid(proposal_classes)

            all_detection_bboxes = []
            all_detection_classes = []
            all_detection_probs = []
            all_detection_batch_indices = []
            all_detection_vertices = []

            for batch_index in range(batch_size):
                if self.iteration:
                    class_bboxes = detection_bboxes2[batch_index, :, 0, :]
                    class_proposal_bboxes = detection_bboxes[batch_index,:,0,:]
                else:
                    class_bboxes = detection_bboxes[batch_index, :, 0, :]
                    class_proposal_bboxes = proposal_bboxes[batch_index,:,:]
                class_probs = detection_probs[batch_index, :, 0]
                class_vertices = proposal_vertices[batch_index]
                threshold = 0.3
                kept_indices = nms(class_bboxes, class_probs, threshold)

                class_bboxes = class_bboxes[kept_indices]
                class_probs = class_probs[kept_indices]
                class_vertices = class_vertices[kept_indices].view(-1,2,8)
                class_proposal_bboxes = class_proposal_bboxes[kept_indices]

                final_vertices = torch.empty(class_vertices.size(), dtype=torch.float, device=class_vertices.device)
                bboxes_centers = torch.stack(( (class_proposal_bboxes[:,0] + class_proposal_bboxes[:,2])/2,(class_proposal_bboxes[:,1] + class_proposal_bboxes[:,3])/2), dim=1)
                width = class_proposal_bboxes[:,2] - class_proposal_bboxes[:,0]
                height = class_proposal_bboxes[:,3] - class_proposal_bboxes[:,1]
                for i in range(class_vertices.size()[-1]):
                    final_vertices[:,:,i] = torch.stack((class_vertices[:,0,i]*width + bboxes_centers[:,0], class_vertices[:,1,i]*height + bboxes_centers[:,1]), dim=1)

                all_detection_bboxes.append(class_bboxes)
                all_detection_classes.append(torch.full((len(kept_indices),), 0, dtype=torch.int))
                all_detection_probs.append(class_probs)
                all_detection_batch_indices.append(torch.full((len(kept_indices),), batch_index, dtype=torch.long))
                all_detection_vertices.append(final_vertices)

            all_detection_bboxes = torch.cat(all_detection_bboxes, dim=0)
            all_detection_classes = torch.cat(all_detection_classes, dim=0)
            all_detection_probs = torch.cat(all_detection_probs, dim=0)
            all_detection_batch_indices = torch.cat(all_detection_batch_indices, dim=0)
            all_detection_vertices = torch.cat(all_detection_vertices, dim=0)
            return all_detection_bboxes, all_detection_classes, all_detection_probs, all_detection_vertices, all_detection_batch_indices
