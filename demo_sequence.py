import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import cv2
from smplx import SMPLHLayer, SMPLXLayer

from stackflow.configs import load_config
from stackflow.datasets.intercap_hoi_dataset import InterCapDataset
from stackflow.datasets.behave_extend_hoi_dataset import BEHAVEExtendDataset
from stackflow.datasets.intercap_metadata import InterCapMetaData
from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from stackflow.datasets.utils import load_pickle, save_pickle
from stackflow.utils.visualize import render_hoi
from stackflow.models import Model
from stackflow.utils.utils import to_device, set_seed
from stackflow.utils.optimization_sequence import post_optimization_sequence


class SeqDataset():

    def __init__(self, img_ids, dataset):
        self.dataset = dataset
        self.img_ids = img_ids


    def __len__(self, ):
        return len(self.img_ids)


    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        return self.dataset.load_item(img_id)


def inference(args, cfg):
    device = torch.device('cuda')

    model = Model(cfg)
    model.to(device)
    print('Loading checkpoint from {}.'.format(cfg.eval.checkpoint))
    model.load_checkpoint(cfg.eval.checkpoint)
    model.eval()

    if cfg.dataset.name == 'InterCap':
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

    seq_name, cam_idx = args.seq_id.split('.')
    try:
        image_ids = image_list_by_seq[seq_name][int(cam_idx)]
    except:
        image_ids = image_list_by_seq[seq_name][str(cam_idx)]
    reconstruction_results = {}
    print('begin to evaluate sequence: {} (cam {}) with length {}'.format(seq_name, str(cam_idx), str(len(image_ids))))
    dataset = SeqDataset(image_ids, test_dataset)
    test_dataloader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size=cfg.eval.batch_size,
                                                   num_workers=cfg.eval.num_workers,
                                                   shuffle=False,
                                                   drop_last=False)
    all_batch, all_pred = {}, {}
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

    output_dir = 'outputs/demo/sequences/'
    os.makedirs(output_dir, exist_ok=True)
    save_pickle(reconstruction_results, os.path.join(output_dir, '{}_recon_results_sequencewise_with_post_optim.pkl'.format(args.seq_id)))


def visualize_sequence(args, cfg):
    device = torch.device('cuda')
    if cfg.dataset.name == 'InterCap':
        metadata = InterCapMetaData(cfg.dataset.root_dir)
        smpl = SMPLXLayer(model_path='data/models/smplx', gender='neutral').to(device)
    elif cfg.dataset.name == 'BEHAVE-Extended':
        metadata = BEHAVEExtendMetaData(cfg.dataset.root_dir)
        smpl = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
    else:
        raise ValueError('Unsupported dataset {}.'.format(cfg.dataset.name))

    output_dir = 'outputs/demo/sequences/'

    reconstruction_results = load_pickle(os.path.join(output_dir, '{}_recon_results_sequencewise_with_post_optim.pkl'.format(args.seq_id)))

    for seq_id in reconstruction_results:
        if cfg.dataset.name == 'InterCap':
            obj_name = seq_id.split('_')[1]
            obj_name = metadata.OBJECT_IDX2NAME[obj_name]
        elif cfg.dataset.name == 'BEHAVE-Extended':
            obj_name = seq_id.split('_')[2]
        cam_id = int(seq_id.split('_')[-1])
        img_ids = reconstruction_results[seq_id]['img_id']
        smpl_betas = reconstruction_results[seq_id]['betas']
        smpl_body_pose_rotmat = reconstruction_results[seq_id]['body_pose_rotmat']
        hoi_trans = reconstruction_results[seq_id]['hoi_trans']
        hoi_rotmat = reconstruction_results[seq_id]['hoi_rotmat']
        obj_rel_R = reconstruction_results[seq_id]['obj_rel_R']
        obj_rel_T = reconstruction_results[seq_id]['obj_rel_T']

        smpl_betas = torch.tensor(smpl_betas, dtype=torch.float32).to(device)
        smpl_body_pose_rotmat = torch.tensor(smpl_body_pose_rotmat, dtype=torch.float32).to(device)
        smpl_out = smpl(betas=smpl_betas, body_pose=smpl_body_pose_rotmat)
        smpl_v = smpl_out.vertices.detach()
        smpl_J = smpl_out.joints.detach()
        smpl_v = smpl_v - smpl_J[:, 0:1]

        obj_rel_R = torch.tensor(obj_rel_R, dtype=torch.float32).to(device).reshape(-1, 3, 3)
        obj_rel_T = torch.tensor(obj_rel_T, dtype=torch.float32).to(device).reshape(-1, 1, 3)
        hoi_rotmat = torch.tensor(hoi_rotmat, dtype=torch.float32).to(device).reshape(-1, 3, 3)
        hoi_trans = torch.tensor(hoi_trans, dtype=torch.float32).to(device).reshape(-1, 1, 3)
        smpl_v = torch.matmul(smpl_v, hoi_rotmat.permute(0, 2, 1)) + hoi_trans
        smpl_v = smpl_v.cpu().numpy()
        smpl_f = smpl.faces.astype(np.int64)

        object_v, object_f = metadata.obj_mesh_templates[obj_name]
        object_v = torch.tensor(object_v, dtype=torch.float32).to(device).reshape(1, -1, 3)
        object_v = torch.matmul(object_v, obj_rel_R.permute(0, 2, 1)) + obj_rel_T
        object_v = torch.matmul(object_v, hoi_rotmat.permute(0, 2, 1)) + hoi_trans
        object_v = object_v.cpu().numpy()

        if cfg.dataset.name == 'InterCap':
            calitration = metadata.cam_calibration[str(cam_id)]
            cx, cy = calitration['c']
            fx, fy = calitration['f']
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        elif cfg.dataset.name == 'BEHAVE-Extended':
            cx, cy, fx, fy = metadata.cam_intrinsics[cam_id]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        w, h = metadata.IMG_WIDTH, metadata.IMG_HEIGHT
        if args.side_view:
            video = cv2.VideoWriter(os.path.join(output_dir, '{}_recon_results_sequencewise_with_post_optim_side_view.mp4'.format(args.seq_id)), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        else:
            video = cv2.VideoWriter(os.path.join(output_dir, '{}_recon_results_sequencewise_with_post_optim.mp4'.format(args.seq_id)), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for idx, img_id in enumerate(tqdm(img_ids, desc='render images')):
            if args.side_view:
                image = np.ones((h, w, 3), dtype=np.uint8) * 255

                rot = -90
                _smpl_v = rotation(smpl_v[idx], smpl_v[idx].mean(0), rot)
                _object_v = rotation(object_v[idx], smpl_v[idx].mean(0), rot)
                image = render_hoi(image, _smpl_v, smpl_f, _object_v, object_f, K)
            else:
                image = cv2.imread(metadata.get_image_path(img_id))
                image = render_hoi(image, smpl_v[idx], smpl_f, object_v[idx], object_f, K)
            video.write(image.astype(np.uint8))
        video.release()


def rotation(v, center, angle):
    v_centered = v - center.reshape(1, 3)
    from scipy.spatial.transform import Rotation
    rot_matrix = Rotation.from_euler('y', angle, degrees=True).as_matrix()
    v_out = v_centered @ rot_matrix + center.reshape(1, 3)
    return v_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', default='stackflow/configs/behave.yaml', type=str)
    parser.add_argument('--dataset_root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str)
    parser.add_argument('--seq_id', default="Date03_Sub03_backpack_back.0", type=str)
    parser.add_argument('--side_view', default=False, action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    cfg.dataset.root_dir = args.dataset_root_dir
    cfg.freeze()

    inference(args, cfg)
    visualize_sequence(args, cfg)
