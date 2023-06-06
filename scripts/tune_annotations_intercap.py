import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import trimesh
from tqdm import tqdm
import numpy as np
import argparse

import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.transforms import matrix_to_rotation_6d, axis_angle_to_matrix, rotation_6d_to_matrix, matrix_to_axis_angle

from stackflow.datasets.utils import load_pickle, save_pickle
from stackflow.datasets.intercap_metadata import InterCapMetaData


def tune(annotation, smplx_v, object_v, object_template_v, smplx_model):
    device = torch.device('cuda')
    object_gt_v = torch.tensor(object_v, dtype=torch.float32).to(device).reshape(1, -1, 3)
    smplx_gt_v = torch.tensor(smplx_v, dtype=torch.float32).to(device).reshape(1, -1, 3)
    object_template_v = torch.tensor(object_template_v, dtype=torch.float32).to(device).reshape(1, -1, 3)

    object_pose_init = torch.tensor(annotation['ob_pose'], dtype=torch.float32)
    object_R6d = nn.Parameter(matrix_to_rotation_6d(axis_angle_to_matrix(object_pose_init)).reshape(1, 6).to(device))
    object_trans = nn.Parameter(torch.tensor(annotation['ob_trans'], dtype=torch.float32).reshape(1, 1, 3).to(device))

    smplx_params_init = {}
    for k, v in annotation.items():
        if 'ob_' not in k:
            smplx_params_init[k] = torch.tensor(v)
    smplx_params = {}
    for k, v in smplx_params_init.items():
        smplx_params[k] = nn.Parameter(v.to(device))

    params = list(smplx_params.values())
    params.append(object_R6d)
    params.append(object_trans)

    optimizer = torch.optim.Adam(params, lr=2e-2, betas=(0.9, 0.999))

    iterations = 2
    steps_per_iter = 200
    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        for i in loop:
            optimizer.zero_grad()

            object_v = object_template_v @ rotation_6d_to_matrix(object_R6d).transpose(1, 2) + object_trans

            loss_obj = F.l1_loss(object_v, object_gt_v, reduction='mean')

            smplx_out = smplx_model(return_verts=True, **smplx_params)
            loss_smplx = F.l1_loss(smplx_out.vertices, smplx_gt_v, reduction='mean')

            loss = loss_obj + loss_smplx
            loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            l_str += ', {}: {:0.4f}'.format('loss_obj', loss_obj.item())
            l_str += ', {}: {:0.4f}'.format('loss_smplx', loss_smplx.item())
            loop.set_description(l_str)
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.1

    object_pose_tuned = matrix_to_axis_angle(rotation_6d_to_matrix(object_R6d)).detach().cpu().numpy().reshape(3, )
    object_trans_tuned = object_trans.detach().cpu().numpy().reshape(3, )
    annotation_tuned = {k: v.detach().cpu().numpy() for k, v in smplx_params.items()}
    annotation_tuned['ob_pose'] = object_pose_tuned
    annotation_tuned['ob_trans'] = object_trans_tuned

    return annotation_tuned


def main(args):

    device = torch.device('cuda')
    intercap_metadata = InterCapMetaData(args.root_dir)
    object_templates = intercap_metadata.load_object_mesh_templates()

    smplx_model_male = smplx.create('./data/models', model_type='smplx',
                         gender='male', ext='npz',
                         num_pca_comps=12,
                         create_global_orient=True,
                         create_body_pose=True,
                         create_betas=True,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=True,
                         create_jaw_pose=True,
                         create_leye_pose=True,
                         create_reye_pose=True,
                         create_transl=True).to(device)

    smplx_model_female = smplx.create('./data/models', model_type='smplx',
                         gender='female', ext='npz',
                         num_pca_comps=12,
                         create_global_orient=True,
                         create_body_pose=True,
                         create_betas=True,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=True,
                         create_jaw_pose=True,
                         create_leye_pose=True,
                         create_reye_pose=True,
                         create_transl=True).to(device)
    print('collect all frames...')
    all_seq_ids = list(intercap_metadata.go_through_all_sequences(split='all', target_sub_ids=['06',]))
    print('total {} sequences'.format(len(all_seq_ids)))
    for seq_full_id in tqdm(all_seq_ids, desc='Tune Annotations'):
        seq_dir = intercap_metadata.get_sequence_dir(seq_full_id)

        seq_dir = seq_dir.replace('RGBD_Images', 'Res')
        smpl_annotations = load_pickle(os.path.join(seq_dir, 'res_1.pkl'))
        obj_annotations = load_pickle(os.path.join(seq_dir, 'res_2.pkl'))
        out_seq_dir = seq_dir.replace('Res', 'Res_tuned')
        os.makedirs(out_seq_dir, exist_ok=True)
        annotations_out_files = os.path.join(out_seq_dir, 'res_tuned.pkl')
        if os.path.exists(annotations_out_files) and not args.redo:
            continue

        annotations_tuned = []

        sub_id, obj_id, seq_id = seq_full_id.split('_')
        for frame_name in sorted(os.listdir(os.path.join(seq_dir, 'Mesh'))):
            if '_obj.ply' not in frame_name:
                continue
            object_name = intercap_metadata.OBJECT_IDX2NAME[obj_id]

            gender = intercap_metadata.SUBID_GENDER[sub_id]
            frame_id = int(frame_name.split('_')[0])
            smplx_mesh = trimesh.load(os.path.join(seq_dir, 'Mesh', frame_name.replace('_obj', '')), process=False)
            object_mesh = trimesh.load(os.path.join(seq_dir, 'Mesh', frame_name), process=False)
            smplx_v = np.array(smplx_mesh.vertices)
            object_v = np.array(object_mesh.vertices)
            object_template_v, _ = object_templates[object_name]
            annotation = {
                k: v for k, v in smpl_annotations[frame_id].items() if 'camera' not in k and k != 'pose_embedding'
            }
            annotation['ob_pose'] = obj_annotations['ob_pose'][frame_id]
            annotation['ob_trans'] = obj_annotations['ob_trans'][frame_id]

            if gender == 'male':
                smplx_model = smplx_model_male
            else:
                smplx_model = smplx_model_female
            annotation = tune(annotation, smplx_v, object_v, object_template_v.copy(), smplx_model)

            annotations_tuned.append(annotation)

        save_pickle(annotations_tuned, annotations_out_files)
        print('{}, Done!'.format(annotations_out_files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/storage/data/huochf/InterCap/', type=str, help='Dataset root directory.')
    parser.add_argument('--redo', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
