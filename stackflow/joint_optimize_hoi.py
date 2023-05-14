import os
import json
import trimesh
import math
import cv2
import numpy as np
from typing import Dict
from easydict import EasyDict
import yaml
import argparse
import pickle
import torch

from stackflow.models.rel_dist_decoder import DistanceDecoder
from stackflow.models.hoi_instance_flow import HOIInstance
from stackflow.models.model import Model
from stackflow.models.hoi_decoder import HOIDecoder

from stackflow.datasets.behave_hoi_img_dataset import BEHAVEImageDataset
from stackflow.datasets.intercap_hoi_img_dataset import IntercapImageDataset
from stackflow.datasets.behave_metadata import get_gt_mesh as get_behave_gt_mesh
from stackflow.datasets.intercap_metadata import get_gt_mesh as get_intercap_gt_mesh
from stackflow.datasets.data_utils import save_json, to_device
from stackflow.utils.evaluator import MeshEvaluator, ReconEvaluator
from stackflow.utils.optimization import joint_optimize
from stackflow.utils.optimize_loss import SMPLPriorLoss, RelativeDistanceLoss, PersonKeypointLoss, ObjectReprojLoss
from stackflow.utils.optimize_loss import SMPLPostperioriLoss, RelDistPostperioriLoss, RelativeDistanceFlowLoss


def main(cfg):

    device = torch.device('cuda')

    if cfg.dataset.name == 'behave':
        test_dataset = BEHAVEImageDataset(cfg, exlusion_occlusion=True, is_train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=4, shuffle=True, drop_last=False)
        get_gt_mesh = get_behave_gt_mesh
    else:
        test_dataset = IntercapImageDataset(cfg, exlusion_occlusion=True, is_train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=4, shuffle=False, drop_last=False)
        get_gt_mesh = get_intercap_gt_mesh

    model = Model(cfg)
    model.to(device)

    checkpoint_path = os.path.join(cfg.train.checkpoint_out_dir, '{}.pth'.format(cfg.train.exp))
    if not os.path.exists(checkpoint_path):
        return
    model.load_checkpoint(checkpoint_path)
    model.eval()
    print('Loaded model from {}.'.format(checkpoint_path))

    hoi_decoder = HOIDecoder(smpl_anchor_indices=model.rel_dist_decoder.smpl_anchor_indices, 
                             object_anchor_indices=model.rel_dist_decoder.object_anchor_indices,
                             dataset=cfg.dataset.name,
                             fix_person=False,
                             iterations=0,
                             steps_per_iter=0,
                             lr=0.01,
                             device=device)

    evaluator = MeshEvaluator()
    evaluator_chore = ReconEvaluator(align_mesh=True, smpl_only=False)

    if cfg.dataset.name == 'behave':
        person_keypoints_path = 'data/behave/person_keypoints_test.json'
        with open(person_keypoints_path, 'r') as f:
            person_keypoints = json.load(f)
        object_coor_path = 'data/behave/epro_pnp_behave_recon_results.pkl'
        with open(object_coor_path, 'rb') as f:
            object_coor = pickle.load(f)
    else:
        person_keypoints_path = 'data/intercap/person_keypoints_test.json'
        with open(person_keypoints_path, 'r') as f:
            person_keypoints = json.load(f)
        object_coor_path = 'data/intercap/epro_pnp_intercap_recon_results.pkl'
        with open(object_coor_path, 'rb') as f:
            object_coor = pickle.load(f)

    eval_metrics = {}
    recon_results = {}
    count = 0
    from timeit import default_timer as timer
    begin_time = timer()
    for idx, batch in enumerate(test_dataloader):
        batch = to_device(batch, device)

        outputs = model.inference_step(batch)
        relative_distance = outputs['rel_dist_recon'][:, 0]
        smpl_trans = outputs['translation']
        smpl_betas = outputs['pred_betas']
        smpl_pose_rotmat = outputs['pred_pose_rotmat'][:, 0]
        smpl_params = {
            'translation': smpl_trans,
            'betas': smpl_betas,
            'global_orient': smpl_pose_rotmat[:, :1],
            'body_pose': smpl_pose_rotmat[:, 1:],
        }
        object_labels = batch['object_label']
        hoi_instance_init = hoi_decoder.forward(relative_distance, object_labels, smpl_params)
        hoi_instance_init_out = hoi_instance_init()
        hoi_instance = HOIInstance(hoi_flow=model.hoi_flow, 
                                   object_labels=object_labels, 
                                   condition_features=outputs['conditioning_feats'],
                                   smpl_trans=hoi_instance_init_out['smpl_trans'], 
                                   object_T_relative=hoi_instance_init_out['object_relative_T'], 
                                   object_R_relative=hoi_instance_init_out['object_relative_R'], 
                                   smpl_type='smplh' if cfg.dataset.name == 'behave' else 'smplx', 
                                   dataset=cfg.dataset.name, 
                                   fix_person=False, 
                                   device=device)
        hoi_instance.to(device)

        img_ids = batch['img_id']
        try:
            keypoints = torch.from_numpy(np.array([person_keypoints[img_id]['keypoints'] for img_id in img_ids], dtype=np.float32))
            confidences = torch.from_numpy(np.array([person_keypoints[img_id]['confidence'] for img_id in img_ids], dtype=np.float32))

            obj_x2d = torch.from_numpy(np.array([object_coor[img_id]['x2d'] for img_id in img_ids], dtype=np.float32))
            obj_x3d = torch.from_numpy(np.array([object_coor[img_id]['x3d'] for img_id in img_ids], dtype=np.float32))
            obj_w2d = torch.from_numpy(np.array([object_coor[img_id]['w2d'] for img_id in img_ids], dtype=np.float32))
        except:
            continue
        loss_functions = [
            RelativeDistanceFlowLoss(model.rel_dist_decoder, model.rel_dist_decoder.smpl_anchor_indices, model.rel_dist_decoder.object_anchor_indices[object_labels], object_labels).to(device), 
            PersonKeypointLoss(keypoints, confidences, batch['K'], smpl_type='smplh' if cfg.dataset.name == 'behave' else 'smplx').to(device),
            ObjectReprojLoss(obj_x3d, obj_x2d, obj_w2d, batch['K'], rescaled=True if cfg.dataset.name == 'behave' else False).to(device),
            SMPLPostperioriLoss().to(device),
            RelDistPostperioriLoss().to(device),
        ]
        loss_weights = {
            'rel_dist_loss': lambda cst, it: 10. ** 0 * cst / (1 + it),
            'object_reproj_loss': lambda cst, it: 10. ** 0 * cst / (1 + it),
            'person_keypoints_reproj_loss': lambda cst, it: 10. ** -1 * cst / (1 + it),
            'smpl_postperiori_loss': lambda cst, it: 10. ** 0 * cst / (1 + it),
            'hoi_postperiori_loss': lambda cst, it: 10. ** -1 * cst / (1 + it),
        }

        joint_optimize(hoi_instance, loss_functions, loss_weights, iterations=2, steps_per_iter=300, lr=0.01) # 300

        hoi_instance_out = hoi_instance()
        batch_size = relative_distance.shape[0]
        for b in range(batch_size):
            try:
                gt_smpl_mesh, gt_object_mesh = get_gt_mesh(batch['img_id'][b])
            except:
                # gt mesh may not exists for intercap dataset
                continue
            gt_smpl_vertices = np.array(gt_smpl_mesh.vertices)
            gt_object_vertices = np.array(gt_object_mesh.vertices)
            num_obj_v = gt_object_vertices.shape[0]

            eval_metrics[batch['img_id'][b]] = {}

            recon_smpl_vertices = hoi_instance_init_out['smpl_v'][b].detach().cpu().numpy()
            recon_object_vertices = hoi_instance_init_out['object_v'][b].detach().cpu().numpy()[:num_obj_v]

            smpl_metrics = evaluator.eval_pointcloud(recon_smpl_vertices, gt_smpl_vertices)
            object_metrics = evaluator.eval_pointcloud(recon_object_vertices, gt_object_vertices)
            smpl_chamfer_distance, object_chamfer_distance = evaluator_chore.compute_errors([gt_smpl_mesh, gt_object_mesh], 
                [trimesh.Trimesh(recon_smpl_vertices, gt_smpl_mesh.faces, process=False), trimesh.Trimesh(recon_object_vertices, gt_object_mesh.faces, process=False)])

            smpl_metrics['chamfer_distance_chore'] = smpl_chamfer_distance
            object_metrics['chamfer_distance_chore'] = object_chamfer_distance
            print('Init: {} / {}'.format(idx, len(test_dataloader)), batch['img_id'][b], smpl_metrics['chamfer-L1'], object_metrics['chamfer-L1'], smpl_chamfer_distance, object_chamfer_distance)
            eval_metrics[batch['img_id'][b]]['object_init'] = object_metrics
            eval_metrics[batch['img_id'][b]]['smpl_init'] = smpl_metrics

            recon_smpl_vertices = hoi_instance_out['smpl_v'][b].detach().cpu().numpy()
            recon_object_vertices = hoi_instance_out['object_v'][b].detach().cpu().numpy()[:num_obj_v]

            smpl_metrics = evaluator.eval_pointcloud(recon_smpl_vertices, gt_smpl_vertices)
            object_metrics = evaluator.eval_pointcloud(recon_object_vertices, gt_object_vertices)
            smpl_chamfer_distance, object_chamfer_distance = evaluator_chore.compute_errors([gt_smpl_mesh, gt_object_mesh], 
                [trimesh.Trimesh(recon_smpl_vertices, gt_smpl_mesh.faces, process=False), trimesh.Trimesh(recon_object_vertices, gt_object_mesh.faces, process=False)])

            smpl_metrics['chamfer_distance_chore'] = smpl_chamfer_distance
            object_metrics['chamfer_distance_chore'] = object_chamfer_distance
            print('Joint Optim: {} / {}'.format(idx, len(test_dataloader)), batch['img_id'][b], smpl_metrics['chamfer-L1'], object_metrics['chamfer-L1'], smpl_chamfer_distance, object_chamfer_distance)
            eval_metrics[batch['img_id'][b]]['object_joint_optim'] = object_metrics
            eval_metrics[batch['img_id'][b]]['smpl_joint_optim'] = smpl_metrics

            recon_results[batch['img_id'][b]] = {
                'smpl_betas': hoi_instance_out['smpl_betas'][b].detach().cpu().numpy().tolist(),
                'smpl_body_pose': hoi_instance_out['smpl_body_pose'][b].detach().cpu().numpy().tolist(),
                'smpl_global_orient': hoi_instance_out['smpl_global_orient'][b].detach().cpu().numpy().tolist(),
                'smpl_transl': hoi_instance_out['smpl_trans'][b].detach().cpu().numpy().tolist(),
                'object_R': hoi_instance_out['object_global_rotmat'][b].detach().cpu().numpy().tolist(),
                'object_T': hoi_instance_out['object_global_trans'][b].detach().cpu().numpy().tolist(),
                'smpl_betas_init': hoi_instance_init_out['smpl_betas'][b].detach().cpu().numpy().tolist(),
                'smpl_body_pose_init': hoi_instance_init_out['smpl_body_pose'][b].detach().cpu().numpy().tolist(),
                'smpl_global_orient_init': hoi_instance_init_out['smpl_global_orient'][b].detach().cpu().numpy().tolist(),
                'smpl_transl_init': hoi_instance_init_out['smpl_trans'][b].detach().cpu().numpy().tolist(),
                'object_R_init': hoi_instance_init_out['object_global_rotmat'][b].detach().cpu().numpy().tolist(),
                'object_T_init': hoi_instance_init_out['object_global_trans'][b].detach().cpu().numpy().tolist(),
            }

    end_time = timer()
    print((end_time - begin_time) / count)

    avg_dict = {}
    for img_id, metric_dict in eval_metrics.items():
        for method, metric in metric_dict.items():
            if method not in avg_dict:  
                avg_dict[method] = {}
            for k, v in metric.items():
                if k not in avg_dict[method]:
                    avg_dict[method][k] = v / len(eval_metrics)
                else:
                    avg_dict[method][k] += v / len(eval_metrics)

    eval_metrics['avg'] = avg_dict
    print(avg_dict)
    with open(os.path.join(cfg.train.checkpoint_out_dir, '{}_joint_optim_eval_metric_chore.json'.format(cfg.train.exp)), 'w') as f:
        json.dump(eval_metrics, f, indent=4)
    with open(os.path.join(cfg.train.checkpoint_out_dir, '{}_joint_optim_recon_results_chore.json'.format(cfg.train.exp)), 'w') as f:
        json.dump(recon_results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/model_behave.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    main(cfg)
