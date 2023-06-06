import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import argparse
import trimesh
import random
import numpy as np
import json
from tqdm import tqdm
from smplx import SMPLHLayer, SMPLXLayer
import torch

from stackflow.configs import load_config
from stackflow.datasets.behave_hoi_dataset import BEHAVEDataset
from stackflow.datasets.intercap_hoi_dataset import InterCapDataset
from stackflow.datasets.behave_extend_hoi_dataset import BEHAVEExtendDataset
from stackflow.datasets.behave_metadata import BEHAVEMetaData
from stackflow.datasets.intercap_metadata import InterCapMetaData
from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from stackflow.datasets.utils import save_pickle, save_json, load_pickle, load_json
from stackflow.models import Model
from stackflow.utils.utils import to_device, set_seed
from stackflow.utils.sequence_evaluator import ReconEvaluator as SeqReconEvaluator
from stackflow.utils.evaluator import ReconEvaluator
from stackflow.utils.optimization_sequence import post_optimization_sequence


def get_recon_meshes(dataset_metadata, seq_id, smpl, reconstruction_results):
    device = torch.device('cpu')

    all_smpl_mesh, all_object_mesh = [], []
    for idx, img_id in enumerate(reconstruction_results['img_id']):
        betas = torch.tensor(reconstruction_results['betas'][idx], dtype=torch.float32).reshape(1, 10).to(device)
        body_pose_rotmat = torch.tensor(reconstruction_results['body_pose_rotmat'][idx], dtype=torch.float32).reshape(1, 21, 3, 3).to(device)

        smpl_out = smpl(betas=betas,
                        body_pose=body_pose_rotmat,
                        global_orient=torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3),
                        transl=torch.zeros(3, dtype=torch.float32, device=device).reshape(1, 3))
        smpl_v = smpl_out.vertices
        smpl_J = smpl_out.joints
        smpl_v = smpl_v - smpl_J[:, 0:1]

        hoi_rotmat = reconstruction_results['hoi_rotmat'][idx]
        hoi_trans = reconstruction_results['hoi_trans'][idx]
        smpl_v = smpl_v.detach().cpu().numpy().reshape(-1, 3)
        smpl_v = np.matmul(smpl_v, hoi_rotmat.T) + hoi_trans.reshape(1, 3)
        smpl_f = smpl.faces

        obj_rel_rotmat = reconstruction_results['obj_rel_R'][idx]
        obj_rel_trans = reconstruction_results['obj_rel_T'][idx]
        obj_rotmat = np.matmul(hoi_rotmat, obj_rel_rotmat)
        obj_trans = np.matmul(hoi_rotmat, obj_rel_trans.reshape(3, 1)).reshape(3, ) + hoi_trans
        object_name = dataset_metadata.parse_object_name(img_id)
        object_v, object_f = dataset_metadata.obj_mesh_templates[object_name]
        object_v = np.matmul(object_v.copy(), obj_rotmat.T) + obj_trans.reshape(1, 3) 

        if np.isnan(smpl_v).any():
            smpl_out = smpl(betas=torch.zeros(10, dtype=torch.float32, device=device).reshape(1, 10),
                            body_pose=torch.eye(3, dtype=torch.float32, device=device).reshape(1, 1, 3, 3).repeat(1, 21, 1, 1),
                            global_orient=torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3),
                            transl=torch.zeros(3, dtype=torch.float32, device=device).reshape(1, 3))
            smpl_v = smpl_out.vertices
            smpl_J = smpl_out.joints
            smpl_v = smpl_v - smpl_J[:, 0:1]
            smpl_v = smpl_v.detach().cpu().numpy().reshape(-1, 3)
        if np.isnan(object_v).any():
            object_v, object_f = dataset_metadata.obj_mesh_templates[object_name]
            object_v = object_v.copy()

        smpl_mesh = trimesh.Trimesh(smpl_v, smpl_f, process=False)
        object_mesh = trimesh.Trimesh(object_v, object_f, process=False)

        all_smpl_mesh.append(smpl_mesh)
        all_object_mesh.append(object_mesh)

    return all_smpl_mesh, all_object_mesh


def get_gt_smpl_meshes(smpl, anno):
    device = torch.device('cuda')
    betas = torch.tensor(anno['smplh_betas'], dtype=torch.float32).reshape(1, 10).to(device)
    body_pose_rotmat = torch.tensor(anno['smplh_pose_rotmat'][1:22], dtype=torch.float32).reshape(1, 21, 3, 3).to(device)
    global_orient = torch.tensor(anno['smplh_pose_rotmat'][:1], dtype=torch.float32).reshape(1, 3, 3).to(device)
    transl = torch.tensor(anno['smplh_trans'].astype(np.float32)).reshape(1, 3).to(device)

    smpl_out = smpl(betas=betas,
                    body_pose=body_pose_rotmat,
                    global_orient=global_orient,
                    transl=transl)
    smpl_v = smpl_out.vertices.detach().cpu().numpy().reshape(-1, 3)
    smpl_f = smpl.faces.astype(np.int64)

    return trimesh.Trimesh(smpl_v, smpl_f)



def evaluate(cfg):
    device = torch.device('cpu')

    reconstruction_results_sequencewise = load_pickle(os.path.join('./outputs/stackflow3/intercap/', 'recon_results_sequencewise.pkl'))
    reconstruction_results = load_pickle(os.path.join('./outputs/stackflow3/intercap/', 'recon_results_sequencewise.pkl'))

    smpl = SMPLXLayer(model_path='data/models/smplx', gender='neutral').to(device)

    test_dataset = InterCapDataset(cfg, is_train=False, for_evaluation=True)
    dataset_metadata = test_dataset.dataset_metadata

    hoi_evaluator_seq = SeqReconEvaluator(align_mesh=True, smpl_only=False)
    hoi_evaluator = ReconEvaluator(align_mesh=True, smpl_only=False)
    evaluate_results = {}

    seq_id = '07_06_1_5'
    img_id = '07_06_1_5_00132'
    seq_recon_smpl, seq_recon_object = get_recon_meshes(dataset_metadata, seq_id, smpl, reconstruction_results[seq_id])

    gt_smpl, gt_object = dataset_metadata.get_gt_meshes(img_id)

    smpl_error, object_error = hoi_evaluator.compute_errors([gt_smpl, gt_object], [seq_recon_smpl[132], seq_recon_object[132]])
    smpl_error_seq, object_error_seq = hoi_evaluator_seq.compute_errors([[gt_smpl], [gt_object]], [[seq_recon_smpl[132]], [seq_recon_object[132]]])

    print(smpl_error)
    print(smpl_error_seq)
    print()
    print(object_error)
    print(object_error_seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', default='stackflow/configs/behave.yaml', type=str)
    parser.add_argument('--dataset_root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str)
    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    cfg.dataset.root_dir = args.dataset_root_dir
    cfg.freeze()
    set_seed(7)
    evaluate(cfg)
