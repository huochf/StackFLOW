import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import argparse
import cv2
from tqdm import tqdm
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)

from stackflow.datasets.behave_metadata import BEHAVEMetaData
from stackflow.datasets.intercap_metadata import InterCapMetaData
from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from stackflow.datasets.utils import save_pickle


def render_object_coor_map(mesh_verts, mesh_faces, object_rotmat, object_trans, cx, cy, fx, fy, img_h, img_w):
    h, w = img_h, img_w
    device = 'cuda'
    cam_R = torch.FloatTensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
    cam_T = torch.FloatTensor([[0, 0, 0,]]) 
    cameras = PerspectiveCameras(R=cam_R, T=cam_T,
                                 focal_length=[[fx, fy]], 
                                 principal_point=[[ cx,  cy]],
                                 image_size=[[h, w]],
                                 in_ndc=False,
                                 device=device)
    verts = torch.tensor(mesh_verts, dtype=torch.float32).to(device).reshape(-1, 3)
    mesh_faces = torch.tensor(mesh_faces, dtype=torch.int64).reshape(-1, 3)
    R = torch.tensor(object_rotmat, dtype=torch.float32).to(device).reshape(3, 3)
    T = torch.tensor(object_trans, dtype=torch.float32).to(device).reshape(1, 3)
    verts = verts @ R.transpose(0, 1) + T
    input_data = Meshes(verts=[verts.cpu()], faces=[mesh_faces], ).to(device)

    raster_settings = RasterizationSettings(image_size=[h, w], bin_size=0)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(input_data)
    depth = fragments.zbuf
    depth = depth.reshape(h, w)
    mask = torch.zeros_like(depth)
    mask[depth != -1] = 1

    x = torch.arange(0, w, 1, dtype=depth.dtype, device=depth.device).reshape(1, w, 1).repeat(h, 1, 1)
    y = torch.arange(0, h, 1, dtype=depth.dtype, device=depth.device).reshape(h, 1, 1).repeat(1, w, 1)
    z = depth.reshape(h, w, 1)
    x = x - cx
    y = y - cy
    x = x / fx * z
    y = y / fy * z
    xyz = torch.cat([x, y, z], dim=2)

    xyz = xyz - T.reshape(1, 1, 3)
    xyz = torch.matmul(xyz, R.reshape(1, 3, 3))

    xyz[depth == -1, :] = 0

    mask_full = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    if mask.sum() == 0:
        return mask_full, {}

    indices = torch.nonzero(mask)
    ul, _ = indices.min(0)
    br, _ = indices.max(0)
    u, l = ul.cpu().numpy()
    b, r = br.cpu().numpy()

    box_h, box_w = b - u, r - l

    coor = {
        'u': int(u), 'l': int(l), 'h': int(box_h), 'w': int(box_w),
        'coor': xyz[u:u+box_h, l:l+box_w, :].cpu().numpy().astype(np.float32),
    }
    return mask_full, coor


def render_behave(args):

    behave_metadata = BEHAVEMetaData(args.root_dir)
    object_mesh_templates = behave_metadata.load_object_mesh_templates()

    print('collect all frames...')
    all_img_ids = list(behave_metadata.go_through_all_frames(split='all'))
    print('total {} frames'.format(len(all_img_ids)))
    for img_id in tqdm(all_img_ids, desc='Render coordinate maps (BEHAVE)'):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = behave_metadata.parse_img_id(img_id)
        object_coor_path = behave_metadata.get_object_coor_path(img_id)
        object_mask_path = behave_metadata.get_object_full_mask_path(img_id)
        if os.path.exists(object_coor_path) and not args.redo:
            continue

        object_rotmat, object_trans = behave_metadata.load_object_RT(img_id)
        cam_R, cam_T = behave_metadata.cam_RT_matrix[day_id][int(cam_id)]
        object_trans = np.matmul(cam_R.transpose(), object_trans - cam_T)
        object_rotmat = np.matmul(cam_R.transpose(), object_rotmat)

        cx, cy, fx, fy = behave_metadata.cam_intrinsics[int(cam_id)]
        mesh_verts, mesh_faces = object_mesh_templates[obj_name]

        # resize with 0.5 ratio to save memory and speed up render process
        cx, cy, fx, fy = cx / 2, cy / 2, fx / 2, fy / 2
        img_h, img_w = behave_metadata.IMG_HEIGHT // 2, behave_metadata.IMG_WIDTH // 2
        mask_full, coor_map = render_object_coor_map(mesh_verts, mesh_faces, object_rotmat, object_trans, cx, cy, fx, fy, img_h, img_w)

        os.makedirs(os.path.dirname(object_mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(object_coor_path), exist_ok=True)
        cv2.imwrite(object_mask_path, mask_full)
        save_pickle(coor_map, object_coor_path)


def render_behave_extend(args):
    dataset_metadata = BEHAVEExtendMetaData(args.root_dir)
    object_mesh_templates = dataset_metadata.load_object_mesh_templates()

    print('collect all frames...')
    all_img_ids = list(dataset_metadata.go_through_all_frames(split='all'))
    print('total {} frames'.format(len(all_img_ids)))
    for img_id in tqdm(all_img_ids, desc='Render coordinate maps (BEHAVE-Extended)'):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = dataset_metadata.parse_img_id(img_id)
        object_coor_path = dataset_metadata.get_object_coor_path(img_id)
        object_mask_path = dataset_metadata.get_object_full_mask_path(img_id)
        if os.path.exists(object_coor_path) and not args.redo:
            continue

        object_rotmat, object_trans = dataset_metadata.load_object_RT(img_id)
        cam_R, cam_T = dataset_metadata.cam_RT_matrix[day_id][int(cam_id)]
        object_trans = np.matmul(cam_R.transpose(), object_trans - cam_T)
        object_rotmat = np.matmul(cam_R.transpose(), object_rotmat)

        cx, cy, fx, fy = dataset_metadata.cam_intrinsics[int(cam_id)]
        mesh_verts, mesh_faces = object_mesh_templates[obj_name]

        # resize with 0.5 ratio to save memory and speed up render process
        cx, cy, fx, fy = cx / 2, cy / 2, fx / 2, fy / 2
        img_h, img_w = dataset_metadata.IMG_HEIGHT // 2, dataset_metadata.IMG_WIDTH // 2
        mask_full, coor_map = render_object_coor_map(mesh_verts, mesh_faces, object_rotmat, object_trans, cx, cy, fx, fy, img_h, img_w)

        os.makedirs(os.path.dirname(object_mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(object_coor_path), exist_ok=True)
        cv2.imwrite(object_mask_path, mask_full)
        save_pickle(coor_map, object_coor_path)


def render_intercap(args):
    
    intercap_metadata = InterCapMetaData(args.root_dir)
    object_mesh_templates = intercap_metadata.load_object_mesh_templates()
    cam_calibration = intercap_metadata.load_cam_calibration()
    
    print('collect all frames...')
    all_img_ids = list(intercap_metadata.go_through_all_frames(split='all'))
    print('total {} frames'.format(len(all_img_ids)))
    for img_id in tqdm(all_img_ids, desc='Render coordinate maps (InterCap)'):
        sub_id, obj_id, seq_id, cam_id, frame_id = intercap_metadata.parse_img_id(img_id)
        image_path = intercap_metadata.get_image_path(img_id)
        mask_out_path = intercap_metadata.get_object_full_mask_path(img_id)
        coor_out_path = intercap_metadata.get_object_coor_path(img_id)
        if os.path.exists(coor_out_path) and not args.redo:
            continue

        try:
            object_rotmat, object_trans = intercap_metadata.load_object_RT(img_id)
        except:
            # some images may lack of annotations
            continue

        calitration = cam_calibration[cam_id]
        cam_R = np.array(calitration['R'])
        cam_R = Rotation.from_rotvec(cam_R).as_matrix()
        cam_T = np.array(calitration['T'])
        cx, cy = calitration['c']
        fx, fy = calitration['f']

        object_rotmat = np.matmul(cam_R, object_rotmat)
        object_trans = np.matmul(cam_R, object_trans) + cam_T
        object_name = intercap_metadata.OBJECT_IDX2NAME[obj_id]
        mesh_verts, mesh_faces = object_mesh_templates[object_name]

        # resize with 0.5 ratio to save memory and speed up render process
        cx, cy, fx, fy = cx / 2, cy / 2, fx / 2, fy / 2
        img_h, img_w = intercap_metadata.IMG_HEIGHT // 2, intercap_metadata.IMG_WIDTH // 2
        mask_full, coor_map = render_object_coor_map(mesh_verts, mesh_faces, object_rotmat, object_trans, cx, cy, fx, fy, img_h, img_w)

        os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)
        os.makedirs(os.path.dirname(coor_out_path), exist_ok=True)
        cv2.imwrite(mask_out_path, mask_full)
        save_pickle(coor_map, coor_out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str, help='Dataset root directory.')
    parser.add_argument('--is_behave', default=False, action='store_true', help='Process behave dataset or intercap dataset.')
    parser.add_argument('--behave_extend', default=False, action='store_true', help='Process behave-extended datasset')
    parser.add_argument('--redo', default=False, action='store_true')
    args = parser.parse_args()

    if args.is_behave:
        render_behave(args)
    if args.behave_extend:
        render_behave_extend(args)
    else:
        render_intercap(args)
