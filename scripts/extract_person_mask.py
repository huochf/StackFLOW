import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import argparse
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config

from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from stackflow.datasets.intercap_metadata import InterCapMetaData


IMAGE_SIZE = 640
POINTREND_CONFIG = 'data/weights/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'
POINTREND_MODEL_WEIGHTS = 'data/weights/model_final_ba17b9.pkl' # x101-FPN 3x

def get_pointrend_predictor(min_confidence=0.9, image_format='RGB'):
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(POINTREND_CONFIG)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_confidence
    cfg.MODEL.WEIGHTS = POINTREND_MODEL_WEIGHTS
    cfg.INPUT.FORMAT = image_format
    return DefaultPredictor(cfg)


def main_intercap(args):

    segmenter = get_pointrend_predictor()
    intercap_metadata = InterCapMetaData(args.root_dir)

    print('collect all frames...')
    all_frame_ids = list(intercap_metadata.go_through_all_frames(split='all',))
    print('total {} frames'.format(len(all_frame_ids)))
    for frame_id in tqdm(all_frame_ids, desc='Extrack person mask'):
        image_path = intercap_metadata.get_image_path(frame_id)
        save_path = intercap_metadata.get_person_mask_path(frame_id)
        if os.path.exists(save_path) and not args.redo:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
            instances = segmenter(image)['instances']
            person_mask = instances.pred_masks[instances.pred_classes == 0][0]
            person_mask = person_mask.detach().cpu().numpy().astype(np.uint8) * 255
        except:
            person_mask = np.zeros((1080, 1920)).astype(np.uint8)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, person_mask)


def main_behave(args):

    segmenter = get_pointrend_predictor()
    dataset_metadata = BEHAVEExtendMetaData(args.root_dir)
    print('collect all frames...')
    all_frame_ids = list(dataset_metadata.go_through_all_frames(split='all',))[1740000:]
    print('total {} frames'.format(len(all_frame_ids)))
    for frame_id in tqdm(all_frame_ids, desc='Extract person mask'):
        image_path = dataset_metadata.get_image_path(frame_id)
        save_path = dataset_metadata.get_person_mask_path(frame_id)
        if os.path.exists(save_path) and not args.redo:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
            instances = segmenter(image)['instances']
            person_mask = instances.pred_masks[instances.pred_classes == 0][0]
            person_mask = person_mask.detach().cpu().numpy().astype(np.uint8) * 255
        except:
            person_mask = np.zeros((1080, 1920)).astype(np.uint8)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, person_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/storage/data/huochf/InterCap/', type=str, help='Dataset root directory.')
    parser.add_argument('--redo', default=False, action='store_true')
    parser.add_argument('--behave_extend', default=False, action='store_true', help='Process behave-extended datasset')
    args = parser.parse_args()

    if args.behave_extend:
        main_behave(args)
    else:
        main_intercap(args)
