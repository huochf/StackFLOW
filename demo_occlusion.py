import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np
import torch
import trimesh
from smplx import SMPLHLayer
from pytorch3d.transforms import rotation_6d_to_matrix

from stackflow.configs import load_config
from stackflow.utils.utils import set_seed
from stackflow.models import Model
from stackflow.utils.visualize import render_multi_hoi_video_with_offsets
from stackflow.datasets.utils import load_pickle, load_json, load_J_regressor
from stackflow.models.hoi_instances import HOIInstance
from stackflow.utils.optim_losses import ObjectReprojLoss, PersonKeypointLoss, PosteriorLoss


cam_intrinsics = [1018.952, 779.486, 979.7844, 979.840] # we set this fixed for demo visulization, BEHAVE dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
OBJECT_NAME2IDX = {
    'backpack': 0,
    'basketball': 1,
    'boxlarge': 2,
    'boxlong': 3,
    'boxmedium': 4,
    'boxsmall': 5,
    'boxtiny': 6,
    'chairblack': 7,
    'chairwood': 8,
    'keyboard': 9,
    'monitor': 10,
    'plasticcontainer': 11,
    'stool': 12,
    'suitcase': 13,
    'tablesmall': 14,
    'tablesquare': 15,
    'toolbox': 16,
    'trashbin': 17,
    'yogaball': 18,
    'yogamat': 19,
}


def extract_bbox_from_mask(mask):
    try:
        indices = np.array(np.nonzero(np.array(mask)))
        y1 = np.min(indices[0, :])
        y2 = np.max(indices[0, :])
        x1 = np.min(indices[1, :])
        x2 = np.max(indices[1, :])

        return np.array([x1, y1, x2, y2])
    except:
        return np.zeros(4)


def load_data(img_path, obj_name, device):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_height, img_width, _ = image.shape

    person_mask = cv2.imread(img_path.replace('color', 'person_mask'), cv2.IMREAD_GRAYSCALE)
    mask = (person_mask > 127)
    hoi_bbox_xyxy = extract_bbox_from_mask(mask)

    box_width, box_height = hoi_bbox_xyxy[2:] - hoi_bbox_xyxy[:2]
    box_size = max(box_width, box_height)
    box_size = box_size * 1.2
    box_center_x, box_center_y = (hoi_bbox_xyxy[2:] + hoi_bbox_xyxy[:2]) / 2

    box_size = int(box_size)
    box_center_x, box_center_y = int(box_center_x), int(box_center_y)

    img_cropped = np.zeros((box_size, box_size, 3), dtype=np.float32)
    u = max(int(box_center_y - box_size / 2), 0)
    l = max(int(box_center_x - box_size / 2), 0)
    b = min(int(box_center_y + box_size / 2), img_height)
    r = min(int(box_center_x + box_size / 2), img_width)

    y1 = - min(int(box_center_y - box_size / 2), 0) 
    x1 = - min(int(box_center_x - box_size / 2), 0) 
    img_cropped[ y1 : y1 + b - u, x1 : x1 + r - l, :] = image[u:b, l:r, :]
    img_scaled = cv2.resize(img_cropped, (256, 256), interpolation=cv2.INTER_LINEAR)
    img_scaled = img_scaled[:, :, ::-1].astype(np.float32)
    img_scaled = img_scaled.transpose((2, 0, 1))

    for n_c in range(3):
        img_scaled[n_c, :, :] = (img_scaled[n_c, :, :] - mean[n_c]) / std[n_c]

    img_id = img_path.split('.')[0]
    img_patch = torch.tensor(img_scaled).float().reshape(1, 3, 256, 256).to(device)
    box_size = torch.tensor(box_size).float().reshape(1, 1).to(device)
    box_center = torch.tensor([box_center_x, box_center_y]).float().reshape(1, 2).to(device)
    optical_center = torch.tensor(cam_intrinsics[:2]).float().reshape(1, 2).to(device)
    focal_length = torch.tensor(cam_intrinsics[2:]).float().reshape(1, 2).to(device)
    object_label = torch.tensor(OBJECT_NAME2IDX[obj_name], dtype=torch.int64).reshape(1,).to(device)

    object_mesh = trimesh.load('./data/demo/occlusion/{}.ply'.format(obj_name))
    object_v, object_f = np.array(object_mesh.vertices), np.array(object_mesh.faces)
    object_v = object_v - np.array(object_v).mean(axis=0)
    _object_v = np.zeros((1700, 3))
    _object_v[:object_v.shape[0]] = object_v
    object_v = torch.tensor(_object_v).float().reshape(1, -1, 3).to(device)

    return image, {
        'image': img_patch,
        'box_size': box_size,
        'box_center': box_center,
        'optical_center': optical_center,
        'focal_length': focal_length,
        'object_labels': object_label,
        'object_v': object_v,
    }


def run_demo_occlusion(args):
    device = torch.device('cuda')

    cfg = load_config(args.cfg_file)
    cfg.freeze()
    set_seed(7)

    model = Model(cfg)
    model.to(device)
    print('Loading checkpoint from {}.'.format(cfg.eval.checkpoint))
    model.load_checkpoint(cfg.eval.checkpoint)
    model.eval()

    img_path = args.img_path
    objects = img_path.split('/')[-1].split('_')[2]
    objects = [objects, ]
    image, batch = None, {}
    for obj in objects:
        image, data = load_data(img_path, obj, device)
        for k, v in data.items():
            if k not in batch:
                batch[k] = v
            else:
                batch[k] = torch.cat([batch[k], v], dim=0)
    pred = model.inference(batch)

    b = pred['pred_betas'].shape[0]
    smpl = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
    smpl_out = smpl(betas=pred['pred_betas'],
                    body_pose=pred['pred_smpl_body_pose'],
                    global_orient=torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3).repeat(b, 1, 1),
                    transl=torch.zeros(3, dtype=torch.float32, device=device).reshape(1, 3).repeat(b, 1))

    smpl_v = smpl_out.vertices
    smpl_J = smpl_out.joints
    smpl_v = smpl_v - smpl_J[:, 0:1]
    smpl_J = smpl_J - smpl_J[:, 0:1]

    hoi_rotmat = pred['hoi_rotmat'][0].detach().cpu().numpy().reshape(3, 3)
    hoi_trans = pred['hoi_trans'][0].detach().cpu().numpy().reshape(3)
    smpl_v = smpl_v[0].detach().cpu().numpy().reshape(-1, 3)
    smpl_v = np.matmul(smpl_v, hoi_rotmat.T) + hoi_trans.reshape(1, 3)
    smpl_f = smpl.faces.astype(np.int64)

    smpl_J = smpl_J[0].detach().cpu().numpy().reshape(-1, 3)
    smpl_J = np.matmul(smpl_J, hoi_rotmat.T) + hoi_trans.reshape(1, 3)

    object_v_list = []
    object_f_list = []
    for idx, obj in enumerate(objects):
        obj_rel_rotmat = pred['pred_obj_rel_R'][idx].detach().cpu().numpy().reshape(3, 3)
        obj_rel_trans = pred['pred_obj_rel_T'][idx].detach().cpu().numpy().reshape(3)
        obj_rotmat = np.matmul(hoi_rotmat, obj_rel_rotmat)
        obj_trans = np.matmul(hoi_rotmat, obj_rel_trans.reshape(3, 1)).reshape(3, ) + hoi_trans

        object_mesh = trimesh.load('./data/demo/occlusion/{}.ply'.format(obj), process=False)
        object_v, object_f = np.array(object_mesh.vertices), np.array(object_mesh.faces)
        object_v = object_v - np.array(object_v).mean(axis=0)
        object_v = np.matmul(object_v.copy(), obj_rotmat.T) + obj_trans.reshape(1, 3)

        object_v_list.append(object_v)
        object_f_list.append(object_f)

    K = [[cam_intrinsics[2], 0, cam_intrinsics[0]], [0, cam_intrinsics[3], cam_intrinsics[1]], [0, 0, 1]]

    filename = img_path.split('/')[-1].split('.')[0]
    out_dir = './outputs/demo/occlusion/'
    os.makedirs(out_dir, exist_ok=True)

    render_multi_hoi_video_with_offsets(image, pred['pred_offsets'].detach().cpu().numpy(), objects,
        smpl_v, smpl_J, smpl_f, object_v_list, object_f_list, K, './outputs/demo/occlusion/{}_with_offset_recon_results.mp4'.format(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', default='stackflow/configs/behave.yaml', type=str)
    parser.add_argument('--img_path', default='data/demo/occlusion/3_4_monitor_hand_010_0.color.jpg', type=str)
    args = parser.parse_args()

    run_demo_occlusion(args)
