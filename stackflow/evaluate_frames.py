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
from stackflow.utils.evaluator import ReconEvaluator
from stackflow.utils.optimization import post_optimization


def get_recon_meshes(dataset_metadata, img_id, smpl, recon_result):
    device = torch.device('cuda')
    betas = torch.tensor(recon_result['betas'], dtype=torch.float32).reshape(1, 10).to(device)
    body_pose_rotmat = torch.tensor(recon_result['body_pose_rotmat'], dtype=torch.float32).reshape(1, 21, 3, 3).to(device)

    smpl_out = smpl(betas=betas,
                    body_pose=body_pose_rotmat,
                    global_orient=torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3),
                    transl=torch.zeros(3, dtype=torch.float32, device=device).reshape(1, 3))
    smpl_v = smpl_out.vertices
    smpl_J = smpl_out.joints
    smpl_v = smpl_v - smpl_J[:, 0:1]

    hoi_rotmat = recon_result['hoi_rotmat']
    hoi_trans = recon_result['hoi_trans']
    smpl_v = smpl_v.detach().cpu().numpy().reshape(-1, 3)
    smpl_v = np.matmul(smpl_v, hoi_rotmat.T) + hoi_trans.reshape(1, 3)
    smpl_f = smpl.faces

    obj_rel_rotmat = recon_result['obj_rel_R']
    obj_rel_trans = recon_result['obj_rel_T']
    obj_rotmat = np.matmul(hoi_rotmat, obj_rel_rotmat)
    obj_trans = np.matmul(hoi_rotmat, obj_rel_trans.reshape(3, 1)).reshape(3, ) + hoi_trans
    object_name = dataset_metadata.parse_object_name(img_id)
    object_v, object_f = dataset_metadata.obj_mesh_templates[object_name]
    object_v = np.matmul(object_v.copy(), obj_rotmat.T) + obj_trans.reshape(1, 3) 

    smpl_mesh = trimesh.Trimesh(smpl_v, smpl_f, process=False)
    object_mesh = trimesh.Trimesh(object_v, object_f, process=False)
    return smpl_mesh, object_mesh


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
    device = torch.device('cuda')

    if cfg.dataset.name == 'BEHAVE':
        test_dataset = BEHAVEDataset(cfg, is_train=False, for_evaluation=True)
    elif cfg.dataset.name == 'InterCap':
        test_dataset = InterCapDataset(cfg, is_train=False, for_evaluation=True)
    elif cfg.dataset.name == 'BEHAVE-Extended':
        metadata = BEHAVEExtendMetaData(cfg.dataset.root_dir)
        annotations = load_pickle(cfg.dataset.annotation_file_test)
        annotations = {item['img_id']: item for item in annotations}
        annotations_test = {}
        keyframes = load_pickle('./data/datasets/behave-split-30fps-keyframes.pkl')['test']

        seq_renames = {'Date02_Sub02_monitor_move2': 'Date02_Sub02_monitor_move', 
                       'Date02_Sub02_toolbox_part2': 'Date02_Sub02_toolbox',
                       'Date03_Sub04_boxtiny_part2': 'Date03_Sub04_boxtiny',
                       'Date03_Sub04_yogaball_play2': 'Date03_Sub04_yogaball_play',
                       'Date03_Sub05_chairwood_part2': 'Date03_Sub05_chairwood',
                       'Date04_Sub05_monitor_part2': 'Date04_Sub05_monitor'}
        for path in keyframes:
            seq_name, frame_name, file_name = path.split('/')
            if seq_name in seq_renames:
                seq_name = seq_renames[seq_name]
            day_id, sub_id, obj_name, inter_type = metadata.parse_seq_info(seq_name)
            frame_id = frame_name[2:]
            cam_id = file_name[1]
            img_id = '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, cam_id])
            annotations_test[img_id] = annotations[img_id]

        test_dataset = BEHAVEExtendDataset(cfg, annotations_test, is_train=False, for_evaluation=True)

    else:
        raise ValueError('Unsupported dataset {}.'.format(cfg.dataset.name))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size=cfg.eval.batch_size,
                                                   num_workers=cfg.eval.num_workers,
                                                   shuffle=True,
                                                   drop_last=False)

    model = Model(cfg)
    model.to(device)
    print('Loading checkpoint from {}.'.format(cfg.eval.checkpoint))
    model.load_checkpoint(cfg.eval.checkpoint)
    model.eval()

    print('processing all test images ...')
    recon_results = {}
    for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='inference'):
        # if idx > 0:
        #     break
        batch = to_device(batch, device)
        pred = model.inference(batch, debug=False)

        if cfg.eval.post_optim:
            pred = post_optimization(cfg, test_dataset.dataset_metadata, model, batch, pred)

        batch_size = batch['image'].shape[0]
        for batch_idx in range(batch_size):
            img_id = batch['img_id'][batch_idx]
            recon_results[img_id] = {
                'betas': pred['pred_betas'][batch_idx].detach().cpu().numpy(), # (10, ),
                'body_pose_rotmat': pred['pred_smpl_body_pose'][batch_idx].detach().cpu().numpy(), # (21, 3, 3)
                'hoi_trans': pred['hoi_trans'][batch_idx].detach().cpu().numpy(), # (3, )
                'hoi_rotmat': pred['hoi_rotmat'][batch_idx].detach().cpu().numpy(), # (3, 3)
                'obj_rel_R': pred['pred_obj_rel_R'][batch_idx].detach().cpu().numpy(), # (3, 3)
                'obj_rel_T': pred['pred_obj_rel_T'][batch_idx].detach().cpu().numpy(), # (3, )
            }
    output_dir = cfg.eval.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if cfg.eval.post_optim:
        save_pickle(recon_results, os.path.join(output_dir, 'recon_results_with_post_optim.pkl'))
    else:
        save_pickle(recon_results, os.path.join(output_dir, 'recon_results.pkl'))

    print('calculate chamfer distances ...')
    if cfg.dataset.name == 'BEHAVE':
        smpl = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
    elif cfg.dataset.name == 'InterCap':
        smpl = SMPLXLayer(model_path='data/models/smplx', gender='neutral').to(device)
    elif cfg.dataset.name == 'BEHAVE-Extended':
        smpl = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
        smpl_male = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
        smpl_female = SMPLHLayer(model_path='data/models/smplh', gender='female').to(device)
    else:
        raise ValueError('Unsupported dataset {}.'.format(cfg.dataset.name))
    dataset_metadata = test_dataset.dataset_metadata
    hoi_evaluator = ReconEvaluator(align_mesh=False, smpl_only=False)
    hoi_evaluator_aligned = ReconEvaluator(align_mesh=True, smpl_only=False)
    smpl_evaluator_aligned = ReconEvaluator(align_mesh=True, smpl_only=True)
    evaluate_results = {}
    for img_id in tqdm(recon_results.keys(), desc='evaluate'):
        print(img_id)

        recon_smpl, recon_object = get_recon_meshes(dataset_metadata, img_id, smpl, recon_results[img_id])
        try:
            if cfg.dataset.name == 'BEHAVE-Extended':
                anno = annotations[img_id]
                if anno['gender'] == 'male':
                    gt_smpl = get_gt_smpl_meshes(smpl_male, anno)
                else:
                    gt_smpl = get_gt_smpl_meshes(smpl_female, anno)

                obj_name = img_id.split('_')[2]
                object_v, object_f = dataset_metadata.obj_mesh_templates[obj_name]
                object_r = anno['object_rotmat']
                object_t = anno['object_trans']
                _object_v = np.matmul(object_v.copy(), object_r.T) + object_t.reshape((1, 3))
                gt_object = trimesh.Trimesh(_object_v, object_f)
            else:
                gt_smpl, gt_object = dataset_metadata.get_gt_meshes(img_id)
        except: # gt mesh may not exists
            print('Can not find mesh for image {}.'.format(img_id))
            continue

        try:
            smpl_error, object_error = hoi_evaluator.compute_errors([gt_smpl, gt_object], [recon_smpl, recon_object])
            hoi_smpl_error, hoi_obj_error = hoi_evaluator_aligned.compute_errors([gt_smpl, gt_object], [recon_smpl, recon_object])
            smpl_aligned_error, _ = smpl_evaluator_aligned.compute_errors([gt_smpl, gt_object], [recon_smpl, recon_object])
        except: # Nan values
            print('Exception occurs during calculate errors.')
            continue
        evaluate_results[img_id] = {
            'hoi_smpl_error': hoi_smpl_error,
            'hoi_obj_error': hoi_obj_error,
            'smpl_error': smpl_error,
            'object_error': object_error,
            'smpl_aligned_error': smpl_aligned_error,
        }
        print(evaluate_results[img_id])

    all_hoi_smpl_errors = [item['hoi_smpl_error'] for item in evaluate_results.values()]
    all_hoi_obj_errors = [item['hoi_obj_error'] for item in evaluate_results.values()]
    all_smpl_errors = [item['smpl_error'] for item in evaluate_results.values()]
    all_object_errors = [item['object_error'] for item in evaluate_results.values()]
    all_smpl_aligned_errors = [item['smpl_aligned_error'] for item in evaluate_results.values()]

    evaluate_results['avg'] = {
        'hoi_smpl_error': np.mean(all_hoi_smpl_errors),
        'hoi_obj_error': np.mean(all_hoi_obj_errors),
        'smpl_error': np.mean(all_smpl_errors),
        'object_error': np.mean(all_object_errors),
        'smpl_aligned_error': np.mean(all_smpl_aligned_errors),
    }
    evaluate_results['std'] = {
        'hoi_smpl_error': np.std(all_hoi_smpl_errors),
        'hoi_obj_error': np.std(all_hoi_obj_errors),
        'smpl_error': np.std(all_smpl_errors),
        'object_error': np.std(all_object_errors),
        'smpl_aligned_error': np.std(all_smpl_aligned_errors),
    }
    print(evaluate_results['avg'])

    if cfg.eval.post_optim:
        save_json(evaluate_results, os.path.join(output_dir, 'evaluate_results_with_post_optim.json'))
    else:
        save_json(evaluate_results, os.path.join(output_dir, 'evaluate_results.json'))


def metrics_without_occlusion(cfg):
    output_dir = cfg.eval.output_dir
    if cfg.eval.post_optim:
        evaluate_metrics = load_json(os.path.join(output_dir, 'evaluate_results_with_post_optim.json'))
    else:
        evaluate_metrics = load_json(os.path.join(output_dir, 'evaluate_results.json'))

    if cfg.dataset.name == 'BEHAVE':
        dataset_metadata = BEHAVEMetaData(cfg.dataset.root_dir)
    elif cfg.dataset.name == 'InterCap':
        dataset_metadata = InterCapMetaData(cfg.dataset.root_dir)
    elif cfg.dataset.name == 'BEHAVE-Extended':
        dataset_metadata = BEHAVEExtendMetaData(cfg.dataset.root_dir)
    else:
        raise ValueError('Unsupported dataset {}.'.format(cfg.dataset.name))

    all_hoi_smpl_errors = []
    all_hoi_obj_errors = []
    all_smpl_errors = []
    all_object_errors = []
    all_smpl_aligned_errors = []

    for img_id, metrics in tqdm(evaluate_metrics.items()):
        if img_id == 'avg' or img_id == 'std':
            continue
        visible_ratio = dataset_metadata.get_obj_visible_ratio(img_id)
        # print(visible_ratio)
        if visible_ratio < 0.3 or np.isnan(metrics['hoi_smpl_error']):
            continue
        all_hoi_smpl_errors.append(metrics['hoi_smpl_error'])
        all_hoi_obj_errors.append(metrics['hoi_obj_error'])
        all_smpl_errors.append(metrics['smpl_error'])
        all_object_errors.append(metrics['object_error'])
        all_smpl_aligned_errors.append(metrics['smpl_aligned_error'])

    evaluate_results = {}
    evaluate_results['avg'] = {
        'hoi_smpl_error': np.mean(all_hoi_smpl_errors),
        'hoi_obj_error': np.mean(all_hoi_obj_errors),
        'smpl_error': np.mean(all_smpl_errors),
        'object_error': np.mean(all_object_errors),
        'smpl_aligned_error': np.mean(all_smpl_aligned_errors),
    }
    evaluate_results['std'] = {
        'hoi_smpl_error': np.std(all_hoi_smpl_errors),
        'hoi_obj_error': np.std(all_hoi_obj_errors),
        'smpl_error': np.std(all_smpl_errors),
        'object_error': np.std(all_object_errors),
        'smpl_aligned_error': np.std(all_smpl_aligned_errors),
    }
    print(evaluate_results['avg'])

    if cfg.eval.post_optim:
        save_json(evaluate_results, os.path.join(output_dir, 'evaluate_results_with_post_optim_no_occlusion.json'))
    else:
        save_json(evaluate_results, os.path.join(output_dir, 'evaluate_results_no_occlusion.json'))


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
    metrics_without_occlusion(cfg)
