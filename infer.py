import argparse
import os
import random
import torch
# import json
from tqdm import tqdm
import numpy as np

from PIL import ImageDraw
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from bbox import BBox
from model import Model
from roi.pooler import Pooler
from config.eval_config import EvalConfig as Config


def _infer(path_to_input_image: str, path_to_output_image: str, path_to_checkpoint: str, dataset_name: str, backbone_name: str, prob_thresh: float):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_class = DatasetBase.from_name(dataset_name)
    backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
    model = Model(backbone, dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).to(device)
    model.load(path_to_checkpoint)

    if os.path.isfile(path_to_input_image):
        files = [path_to_input_image]
    else:
        files = os.listdir(path_to_input_image)
        print('Running inference on folder:', path_to_input_image)

    with torch.no_grad():
        for file in tqdm(files):
            image = transforms.Image.open(os.path.join(path_to_input_image, file))
            image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

            detection_bboxes, detection_probs, detection_vertices, _ = \
                model.eval().forward(image_tensor.unsqueeze(dim=0).to(device))
            detection_bboxes /= scale
            detection_vertices /= scale

            kept_indices = detection_probs > prob_thresh
            detection_bboxes = detection_bboxes[kept_indices]
            detection_probs = detection_probs[kept_indices]
            detection_vertices = detection_vertices[kept_indices]

            draw = ImageDraw.Draw(image)

            for bbox, prob, vert in zip(detection_bboxes.tolist(), detection_probs.tolist(), detection_vertices.tolist()):
                color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                category = "cuboid"

                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
                draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

                quads = []
                quads.append(((int(vert[0][0]), int(vert[1][0])), (int(vert[0][1]), int(vert[1][1])), (int(vert[0][3]), int(vert[1][3])), (int(vert[0][2]), int(vert[1][2])), (int(vert[0][0]), int(vert[1][0]))))
                quads.append(((int(vert[0][0]), int(vert[1][0])), (int(vert[0][4]), int(vert[1][4])), (int(vert[0][5]), int(vert[1][5])), (int(vert[0][1]), int(vert[1][1])), (int(vert[0][0]), int(vert[1][0]))))
                quads.append(((int(vert[0][0]), int(vert[1][0])), (int(vert[0][4]), int(vert[1][4])), (int(vert[0][6]), int(vert[1][6])), (int(vert[0][2]), int(vert[1][2])), (int(vert[0][0]), int(vert[1][0]))))
                quads.append(((int(vert[0][1]), int(vert[1][1])), (int(vert[0][5]), int(vert[1][5])), (int(vert[0][7]), int(vert[1][7])), (int(vert[0][3]), int(vert[1][3])), (int(vert[0][1]), int(vert[1][1]))))
                quads.append(((int(vert[0][4]), int(vert[1][4])), (int(vert[0][5]), int(vert[1][5])), (int(vert[0][7]), int(vert[1][7])), (int(vert[0][6]), int(vert[1][6])), (int(vert[0][4]), int(vert[1][4]))))
                quads.append(((int(vert[0][2]), int(vert[1][2])), (int(vert[0][3]), int(vert[1][3])), (int(vert[0][7]), int(vert[1][7])), (int(vert[0][6]), int(vert[1][6])), (int(vert[0][2]), int(vert[1][2]))))

                for quad in quads:
                    draw.line(quad, fill=color)

            output_path = os.path.join(path_to_output_image, file)
            image.save(output_path)

            if detection_probs.size()[0] > 0:
                max_index = torch.argmax(detection_probs)
                detection_vertices = detection_vertices[max_index]
            detection_vertices = detection_vertices.cpu().numpy()
            with open(os.path.join(path_to_output_image, file + '.npy'), 'wb') as f:
                np.save(f, detection_vertices)

if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
        parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
        parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('-p', '--probability_threshold', type=float, default=0.6, help='threshold of detection probability')
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
        parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        parser.add_argument('input', type=str, help='path to input image')
        parser.add_argument('output', type=str, help='path to output result image')
        args = parser.parse_args()

        path_to_input_image = args.input
        path_to_output_image = args.output
        dataset_name = args.dataset
        backbone_name = args.backbone
        path_to_checkpoint = args.checkpoint
        prob_thresh = args.probability_threshold

        os.makedirs(os.path.join(os.path.curdir, os.path.dirname(path_to_output_image)), exist_ok=True)

        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_sizes=args.anchor_sizes, pooler_mode=args.pooler_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n)

        # print('Arguments:')
        # for k, v in vars(args).items():
        #     print(f'\t{k} = {v}')
        # print(Config.describe())

        _infer(path_to_input_image, path_to_output_image, path_to_checkpoint, dataset_name, backbone_name, prob_thresh)

    main()
