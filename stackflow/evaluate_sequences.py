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
from stackflow.utils.sequence_evaluator import ReconEvaluator
from stackflow.utils.optimization_sequence import post_optimization_sequence


def get_recon_meshes(dataset_metadata, seq_id, smpl, reconstruction_results):
    device = torch.device('cuda')

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

    return trimesh.Trimesh(smpl_v, smpl_f, process=False)


class SeqDataset():

    def __init__(self, img_ids, dataset):
        self.dataset = dataset
        self.img_ids = img_ids


    def __len__(self, ):
        return len(self.img_ids)


    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        return self.dataset.load_item(img_id)


def evaluate(cfg):
    device = torch.device('cuda')

    model = Model(cfg)
    model.to(device)
    print('Loading checkpoint from {}.'.format(cfg.eval.checkpoint))
    model.load_checkpoint(cfg.eval.checkpoint)
    model.eval()

    if cfg.dataset.name == 'BEHAVE':
        test_dataset = BEHAVEDataset(cfg, is_train=False, for_evaluation=True)
    elif cfg.dataset.name == 'InterCap':
        test_dataset = InterCapDataset(cfg, is_train=False, for_evaluation=True)
    elif cfg.dataset.name == 'BEHAVE-Extended':
        metadata = BEHAVEExtendMetaData(cfg.dataset.root_dir)
        annotations = load_pickle(cfg.dataset.annotation_file_test)
        annotations = {item['img_id']: item for item in annotations}
        test_dataset = BEHAVEExtendDataset(cfg, annotations, is_train=False, for_evaluation=True)

    else:
        raise ValueError('Unsupported dataset {}.'.format(cfg.dataset.name))

    dataset_metadata = test_dataset.dataset_metadata
    image_list_by_seq = dataset_metadata.get_all_image_by_sequence(split='test')

    reconstruction_results = {}
    for seq_name in tqdm(image_list_by_seq):
        for cam_idx in image_list_by_seq[seq_name]:
            image_ids = image_list_by_seq[seq_name][cam_idx]
            print('begin to evaluate sequence: {} (cam {}) with length {}'.format(seq_name, str(cam_idx), str(len(image_ids))))
            dataset = SeqDataset(image_ids, test_dataset)
            test_dataloader = torch.utils.data.DataLoader(dataset, 
                                                           batch_size=cfg.eval.batch_size,
                                                           num_workers=cfg.eval.num_workers,
                                                           shuffle=False,
                                                           drop_last=False)
            all_batch, all_pred = {}, {}
            try:
                for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='inference'):

                    def append(dict1, dict2, keys=[]):
                        for k in dict2:
                            if keys != [] and k not in keys:
                                continue
                            if k not in dict1:
                                if isinstance(dict2[k], torch.Tensor):
                                    dict1[k] = dict2[k].detach().cpu()
                                else:
                                    dict1[k] = dict2[k]
                            else:
                                if isinstance(dict2[k], torch.Tensor):
                                    dict1[k] = torch.cat([dict1[k], dict2[k].detach().cpu()], dim=0)
                                else:
                                    dict1[k].extend(dict2[k])
                        return dict1

                    all_batch = append(all_batch, batch, keys=['img_id', 'optical_center', 'focal_length', 'object_labels', 'obj_x3d', 'obj_x2d', 'obj_w2d', 'person_kps'])

                    batch = to_device(batch, device)
                    pred = model.inference(batch, debug=False)

                    all_pred = append(all_pred, pred, keys=['pred_betas', 'pred_pose6d', 'pred_smpl_body_pose', 'hoi_rotmat', 'hoi_trans', 'pred_obj_rel_T', 'pred_obj_rel_R', 'human_features', 'hoi_features'])
            except:
                print(seq_name, cam_idx)
                print('Exception occurs during loading image.') # annotation may not exists.
                # Date03_Sub03_basketball, Date03_Sub03_boxlarge, Date03_Sub03_keyboard_move, Date03_Sub03_keyboard_typing,
                # Date03_Sub04_basketball, Date03_Sub04_boxlarge, Date03_Sub04_boxtiny, Date03_Sub04_boxtiny, Date03_Sub04_chairblack_hand,
                # Date03_Sub04_keyboard_moveDate03_Sub04_keyboard_typing, Date03_Sub04_monitor_hand, Date03_Sub04_yogaball_play, Date03_Sub04_yogaball_sit
                # Date03_Sub05_basketball, Date03_Sub05_basketball, Date03_Sub05_chairblack, Date03_Sub05_chairwood, Date03_Sub05_keyboard, Date03_Sub05_monitor,
                continue

            if cfg.eval.post_optim:
                all_pred = post_optimization_sequence(cfg, dataset_metadata, model, all_batch, all_pred)

            reconstruction_results['{}_{}'.format(seq_name, str(cam_idx))] = {
                'img_id': all_batch['img_id'],
                'betas': all_pred['pred_betas'].detach().cpu().numpy(), # (10, ),
                'body_pose_rotmat': all_pred['pred_smpl_body_pose'].detach().cpu().numpy(), # (21, 3, 3)
                'hoi_trans': all_pred['hoi_trans'].detach().cpu().numpy(), # (3, )
                'hoi_rotmat': all_pred['hoi_rotmat'].detach().cpu().numpy(), # (3, 3)
                'obj_rel_R': all_pred['pred_obj_rel_R'].detach().cpu().numpy(), # (3, 3)
                'obj_rel_T': all_pred['pred_obj_rel_T'].detach().cpu().numpy(), # (3, )
            }

            del all_batch
            del all_pred

    output_dir = cfg.eval.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if cfg.eval.post_optim:
        save_pickle(reconstruction_results, os.path.join(output_dir, 'recon_results_sequencewise_with_post_optim.pkl'))
    else:
        save_pickle(reconstruction_results, os.path.join(output_dir, 'recon_results_sequencewise.pkl'))

    print('calculate chamfer distances ...')
    if cfg.dataset.name == 'InterCap':
        smpl = SMPLXLayer(model_path='data/models/smplx', gender='neutral').to(device)
    elif cfg.dataset.name == 'BEHAVE-Extended':
        smpl = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
        smpl_male = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
        smpl_female = SMPLHLayer(model_path='data/models/smplh', gender='female').to(device)
    else:
        raise ValueError('Unsupported dataset {}.'.format(cfg.dataset.name))
    dataset_metadata = test_dataset.dataset_metadata
    # hoi_evaluator = ReconEvaluator(align_mesh=False, smpl_only=False)
    hoi_evaluator_aligned_win10 = ReconEvaluator(align_mesh=True, smpl_only=False, window_len=10)
    hoi_evaluator_aligned = ReconEvaluator(align_mesh=True, smpl_only=False)
    # smpl_evaluator_aligned = ReconEvaluator(align_mesh=True, smpl_only=True)
    evaluate_results = {}

    for seq_id in tqdm(reconstruction_results.keys(), desc='evaluate'):
        print(seq_id)
        if int(seq_id.split('_')[-1]) != 0:
            # evaluate sequences with cam_id == 0.
            continue

        seq_recon_smpl, seq_recon_object = get_recon_meshes(dataset_metadata, seq_id, smpl, reconstruction_results[seq_id])

        try:
            seq_gt_smpl, seq_gt_object = [], []
            for img_id in reconstruction_results[seq_id]['img_id']:
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
                    gt_object = trimesh.Trimesh(_object_v, object_f, process=False)
                else:
                    gt_smpl, gt_object = dataset_metadata.get_gt_meshes(img_id)
                seq_gt_smpl.append(gt_smpl)
                seq_gt_object.append(gt_object)
        except: # gt mesh may not exists
            print('Can not find mesh for image {}.'.format(img_id))
            continue

        # smpl_error, object_error = hoi_evaluator.compute_errors([seq_gt_smpl, seq_gt_object], [seq_recon_smpl, seq_recon_object])
        hoi_smpl_error, hoi_obj_error = hoi_evaluator_aligned.compute_errors([seq_gt_smpl, seq_gt_object], [seq_recon_smpl, seq_recon_object])
        hoi_smpl_error_win10, hoi_obj_error_win10 = hoi_evaluator_aligned_win10.compute_errors([seq_gt_smpl, seq_gt_object], [seq_recon_smpl, seq_recon_object])
        # smpl_aligned_error, _ = smpl_evaluator_aligned.compute_errors([seq_gt_smpl, seq_gt_object], [seq_recon_smpl, seq_recon_object])

        evaluate_results[seq_id] = {
            'img_ids': reconstruction_results[seq_id]['img_id'],
            'hoi_smpl_avg_error': np.mean(hoi_smpl_error),
            'hoi_obj_avg_error': np.mean(hoi_obj_error),
            'hoi_smpl_avg_error_win10': np.mean(hoi_smpl_error_win10),
            'hoi_obj_avg_error_win10': np.mean(hoi_obj_error_win10),
            # 'smpl_avg_error': np.mean(smpl_error),
            # 'object_avg_error': np.mean(object_error),
            # 'smpl_aligned_avg_error': np.mean(smpl_aligned_error),

            'hoi_smpl_error': hoi_smpl_error,
            'hoi_obj_error': hoi_obj_error,
            'hoi_smpl_error_win10': hoi_smpl_error_win10,
            'hoi_obj_error_win10': hoi_obj_error_win10,
            # 'smpl_error': smpl_error,
            # 'object_error': object_error,
            # 'smpl_aligned_error': smpl_aligned_error,
        }
        # print('hoi_smpl: {:.4f}, hoi_obj: {:.4f}, smpl: {:.4f}, object: {:.4f}, smpl_aligned: {:.4f}'.format(
        #     evaluate_results[seq_id]['hoi_smpl_avg_error'], evaluate_results[seq_id]['hoi_obj_avg_error'], evaluate_results[seq_id]['smpl_avg_error'], 
        #     evaluate_results[seq_id]['object_avg_error'], evaluate_results[seq_id]['smpl_aligned_avg_error']) )
        print('hoi_smpl: {:.4f}, hoi_obj: {:.4f}'.format(evaluate_results[seq_id]['hoi_smpl_avg_error'], evaluate_results[seq_id]['hoi_obj_avg_error']) )

    all_hoi_smpl_errors = np.concatenate([item['hoi_smpl_error'] for item in evaluate_results.values()]).reshape(-1)
    all_hoi_obj_errors = np.concatenate([item['hoi_obj_error'] for item in evaluate_results.values()]).reshape(-1)
    all_hoi_smpl_win10_errors = np.concatenate([item['hoi_smpl_error_win10'] for item in evaluate_results.values()]).reshape(-1)
    all_hoi_obj_win10_errors = np.concatenate([item['hoi_obj_error_win10'] for item in evaluate_results.values()]).reshape(-1)
    # all_smpl_errors = np.concatenate([item['smpl_error'] for item in evaluate_results.values()]).reshape(-1)
    # all_object_errors = np.concatenate([item['object_error'] for item in evaluate_results.values()]).reshape(-1)
    # all_smpl_aligned_errors = np.concatenate([item['smpl_aligned_error'] for item in evaluate_results.values()]).reshape(-1)

    evaluate_results['avg'] = {
        'hoi_smpl_error': np.mean(all_hoi_smpl_errors),
        'hoi_obj_error': np.mean(all_hoi_obj_errors),
        'hoi_smpl_error_win10': np.mean(all_hoi_smpl_win10_errors),
        'hoi_obj_error_win10': np.mean(all_hoi_obj_win10_errors),
        # 'smpl_error': np.mean(all_smpl_errors),
        # 'object_error': np.mean(all_object_errors),
        # 'smpl_aligned_error': np.mean(all_smpl_aligned_errors),
    }
    evaluate_results['std'] = {
        'hoi_smpl_error': np.std(all_hoi_smpl_errors),
        'hoi_obj_error': np.std(all_hoi_obj_errors),
        'hoi_smpl_error_win10': np.std(all_hoi_smpl_win10_errors),
        'hoi_obj_error_win10': np.std(all_hoi_obj_win10_errors),
        # 'smpl_error': np.std(all_smpl_errors),
        # 'object_error': np.std(all_object_errors),
        # 'smpl_aligned_error': np.std(all_smpl_aligned_errors),
    }
    print(evaluate_results['avg'])

    if cfg.eval.post_optim:
        save_json(evaluate_results, os.path.join(output_dir, 'evaluate_results_sequencewise_no_offset_with_post_optim.json'))
    else:
        save_json(evaluate_results, os.path.join(output_dir, 'evaluate_results_sequencewise.json'))


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
