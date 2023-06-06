import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import argparse
import cv2
import random
import numpy as np
import torch

from smplx import SMPLH, SMPLX

from stackflow.datasets.behave_metadata import BEHAVEMetaData
from stackflow.datasets.intercap_metadata import InterCapMetaData
from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from stackflow.datasets.utils import load_pickle
from stackflow.utils.visualize import render_hoi, draw_smpl_joints, draw_object_keypoints, draw_boxes


def main_behave(args):
    annotations = load_pickle(args.anno_file)
    random.shuffle(annotations)

    if args.behave_extend:
        dataset_metadata = BEHAVEExtendMetaData(args.root_dir)
    else:
        dataset_metadata = BEHAVEMetaData(args.root_dir)
    object_templates = dataset_metadata.load_object_mesh_templates()
    output_dir = 'outputs/visualize_anno/behave'
    if args.behave_extend:
        output_dir += '_extend'
    elif args.for_aug:
        output_dir += '_aug'
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(0, args.show_num):
        annotation = annotations[idx]
        img_id = annotation['img_id']
        try:
            day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        except:
            day_id, sub_id, obj_name, inter_type, frame_id, avatar_id, cam_id = img_id.split('_')

        if args.behave_extend:
            image = cv2.imread(dataset_metadata.get_image_path(annotation['img_id']))
        else:
            image = cv2.imread(dataset_metadata.get_image_path(annotation['img_id'], for_aug=args.for_aug))
        # smplh = SMPLH(model_path='data/models/smplh', gender=annotation['gender'], ext='pkl')
        smplh = SMPLH(model_path='data/models/smplh', gender='male', ext='pkl')

        # seperate
        # person_beta = torch.tensor(annotation['smplh_betas_male'].astype(np.float32)).reshape(1, 10)
        # person_body_pose = torch.tensor(annotation['smplh_theta'].astype(np.float32)[3:66]).reshape(1, 63)
        # person_transl = torch.tensor(annotation['smplh_trans'].astype(np.float32)).reshape(1, 3)
        # person_global_orient = torch.tensor(annotation['smplh_theta'].astype(np.float32)[:3]).reshape(1, 3)
        # smpl_out = smplh(betas=person_beta, body_pose=person_body_pose, global_orient=person_global_orient, transl=person_transl,)

        # smpl_v = smpl_out.vertices.detach().reshape(-1, 3)
        # smpl_f = torch.tensor(smplh.faces.astype(np.int64)).reshape(-1, 3)

        # object_v, object_f = object_templates[obj_name]
        # object_r = annotation['object_rotmat']
        # object_t = annotation['object_trans']
        # object_v = np.matmul(object_v, object_r.T) + object_t.reshape((1, 3))

        # joint
        person_beta = torch.tensor(annotation['smplh_betas_male'].astype(np.float32)).reshape(1, 10)
        person_body_pose = torch.tensor(annotation['smplh_theta'].astype(np.float32)[3:66]).reshape(1, 63)
        person_transl = torch.tensor(annotation['smplh_trans'].astype(np.float32)).reshape(1, 3)
        person_global_orient = torch.tensor(annotation['smplh_theta'].astype(np.float32)[:3]).reshape(1, 3)
        smpl_out = smplh(betas=person_beta, body_pose=person_body_pose, global_orient=0 * person_global_orient, transl=0 * person_transl,)
        J_0 = smpl_out.joints.detach()[0]

        smpl_v = smpl_out.vertices.detach().reshape(-1, 3) - J_0[:1]
        smpl_f = torch.tensor(smplh.faces.astype(np.int64)).reshape(-1, 3)

        object_v, object_f = object_templates[obj_name]
        object_r = annotation['object_rel_rotmat']
        object_t = annotation['object_rel_trans']
        object_v = np.matmul(object_v, object_r.T) + object_t.reshape((1, 3))

        hoi_rotmat = annotation['hoi_rotmat']
        hoi_trans = annotation['hoi_trans']
        smpl_v = np.matmul(smpl_v, hoi_rotmat.T) + hoi_trans.reshape(1, 3)
        object_v = np.matmul(object_v, hoi_rotmat.T) + hoi_trans.reshape(1, 3)

        K = annotation['cam_K']

        num_kps = dataset_metadata.object_num_keypoints[obj_name]
        image = render_hoi(image, smpl_v, smpl_f, object_v, object_f, K)
        image = draw_smpl_joints(image, annotation['smplh_joints_2d'])
        image = draw_object_keypoints(image, annotation['obj_keypoints_2d'][:num_kps], obj_name)
        image = draw_boxes(image, annotation['person_bb_xyxy'], annotation['object_bb_xyxy'], annotation['hoi_bb_xyxy'])

        cv2.imwrite(os.path.join(output_dir, '{}.jpg'.format(img_id)), image)


def main_intercap(args):
    annotations = load_pickle(args.anno_file)
    random.shuffle(annotations)

    dataset_metadata = InterCapMetaData(args.root_dir)
    object_templates = dataset_metadata.load_object_mesh_templates()
    output_dir = 'outputs/visualize_anno/intercap'
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(0, args.show_num):
        annotation = annotations[idx]
        img_id = annotation['img_id']

        sub_id, obj_id, seq_id, cam_id, frame_id = img_id.split('_')
        obj_name = dataset_metadata.OBJECT_IDX2NAME[obj_id]

        image = cv2.imread(dataset_metadata.get_image_path(annotation['img_id'],))
        # smplx = SMPLX(model_path='data/models/smplx', gender=annotation['gender'], ext='pkl')
        smplx = SMPLX(model_path='data/models/smplx', gender='neutral', ext='pkl')

        # seperate
        # person_beta = torch.tensor(annotation['smplx_betas_neutral'].astype(np.float32)).reshape(1, 10)
        # person_body_pose = torch.tensor(annotation['smplh_theta'].astype(np.float32)[3:66]).reshape(1, 63)
        # person_transl = torch.tensor(annotation['smplx_trans'].astype(np.float32)).reshape(1, 3)
        # person_global_orient = torch.tensor(annotation['smplh_theta'].astype(np.float32)[:3]).reshape(1, 3)
        # smpl_out = smplx(betas=person_beta, body_pose=person_body_pose, global_orient=person_global_orient, transl=person_transl,)

        # smpl_v = smpl_out.vertices.detach().reshape(-1, 3)
        # smpl_f = torch.tensor(smplx.faces.astype(np.int64)).reshape(-1, 3)

        # object_v, object_f = object_templates[obj_name]
        # object_r = annotation['object_rotmat']
        # object_t = annotation['object_trans']
        # object_v = np.matmul(object_v, object_r.T) + object_t.reshape((1, 3))

        # joint
        person_beta = torch.tensor(annotation['smplx_betas_neutral'].astype(np.float32)).reshape(1, 10)
        person_body_pose = torch.tensor(annotation['smplh_theta'].astype(np.float32)[3:66]).reshape(1, 63)
        person_transl = torch.tensor(annotation['smplx_trans'].astype(np.float32)).reshape(1, 3)
        person_global_orient = torch.tensor(annotation['smplh_theta'].astype(np.float32)[:3]).reshape(1, 3)
        smpl_out = smplx(betas=person_beta, body_pose=person_body_pose, global_orient=0 * person_global_orient, transl=0 * person_transl,)

        J_0 = smpl_out.joints.detach()[0]

        smpl_v = smpl_out.vertices.detach().reshape(-1, 3) - J_0[:1]
        smpl_f = torch.tensor(smplx.faces.astype(np.int64)).reshape(-1, 3)

        object_v, object_f = object_templates[obj_name]
        object_r = annotation['object_rel_rotmat']
        object_t = annotation['object_rel_trans']
        object_v = np.matmul(object_v, object_r.T) + object_t.reshape((1, 3))

        hoi_rotmat = annotation['hoi_rotmat']
        hoi_trans = annotation['hoi_trans']
        smpl_v = np.matmul(smpl_v, hoi_rotmat.T) + hoi_trans.reshape(1, 3)
        object_v = np.matmul(object_v, hoi_rotmat.T) + hoi_trans.reshape(1, 3)

        K = annotation['cam_K']

        num_kps = dataset_metadata.object_num_keypoints[obj_name]
        image = render_hoi(image, smpl_v, smpl_f, object_v, object_f, K)
        image = draw_smpl_joints(image, annotation['smplx_joints_2d'])
        image = draw_object_keypoints(image, annotation['obj_keypoints_2d'][:num_kps], obj_name)
        image = draw_boxes(image, annotation['person_bb_xyxy'], annotation['object_bb_xyxy'], annotation['hoi_bb_xyxy'])

        cv2.imwrite(os.path.join(output_dir, '{}.jpg'.format(img_id)), image)


def main(args):
    if args.is_behave or args.behave_extend:
        main_behave(args)
    else:
        main_intercap(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str, help='Dataset root directory.')
    parser.add_argument('--anno_file', default='./data/datasets/behave_train_list.pkl', type=str, help='Dataset root directory.')
    parser.add_argument('--show_num', default=10, type=int, help='how many images you want to visualize')
    parser.add_argument('--is_behave', default=False, action='store_true')
    parser.add_argument('--behave_extend', default=False, action='store_true', help='Process behave-extended datasset')
    parser.add_argument('--for_aug', default=False, action='store_true', help='For BEHAVE viewport-free augmented dataset. (Only Valid for BEHAVE dataset)')
    args = parser.parse_args()
    main(args)
