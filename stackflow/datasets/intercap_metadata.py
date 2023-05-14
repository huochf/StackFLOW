import os
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from stackflow.datasets.data_utils import load_json

INTERCAP_DIR = '/storage/data/huochf/InterCap/'
IMG_HEIGHT = 1080
IMG_WIDTH = 1920

OBJECT_NAME2IDX = {
    'suitcase1': 0,
    'skate': 1,
    'sports': 2,
    'umbrella': 3,
    'tennis': 4,
    'suitcase2': 5,
    'chair1': 6,
    'bottle': 7,
    'cup': 8,
    'chair2': 9,
}

OBJECT_IDX2NAME = {v: k for k, v in OBJECT_NAME2IDX.items()}

SMPLX_OPENPOSE_INDICES = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                          8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                          63, 64, 65]

OBJECT_MESH_VERTEX_NUM = {
    'suitcase1': 591,
    'skate': 1069,
    'sports': 998,
    'umbrella': 1267,
    'tennis': 2453,
    'suitcase2': 1521,
    'chair1': 342,
    'bottle': 1267,
    'cup': 1469,
    'chair2': 1002,
}

MAX_OBJECT_VERT_NUM = 2500

def load_object_mesh_templates():
    templates = {}
    for object_name, object_label in OBJECT_NAME2IDX.items():
        object_mesh = os.path.join('data/intercap/objs/{:02d}.ply'.format(object_label + 1))

        object_mesh = trimesh.load(object_mesh, process=False)
        verts = torch.tensor(object_mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(object_mesh.faces, dtype=torch.int64)
        templates[object_name] = (verts, faces)

    return templates


def load_cam_calibration():
    cam_intrinsics_params = {}
    for cam_id in range(6):
        if cam_id == 0:
            cam_params = load_json('data/intercap/calibration_third/Color.json')
        else:
            cam_params = load_json('data/intercap/calibration_third/Color_{}.json'.format(cam_id + 1))
        cam_intrinsics_params[str(cam_id + 1)] = cam_params

    return cam_intrinsics_params


def get_image_path(img_id):
    sub_id, obj_id, seg_id, cam_id, frame_id = img_id.split('_')
    image_path = os.path.join(INTERCAP_DIR, 'RGBD_Images', sub_id, obj_id, 'Seg_{}'.format(seg_id), 'Frames_Cam{}'.format(cam_id), 'color', '{}.jpg'.format(frame_id))
    return image_path


def get_person_mask_path(img_id, visiable=True):
    sub_id, obj_id, seg_id, cam_id, frame_id = img_id.split('_')
    if visiable:
        mask_path = os.path.join(INTERCAP_DIR, '_rendered_mask', sub_id, obj_id, 'Seg_{}'.format(seg_id), 'Frames_Cam{}'.format(cam_id), '{:05d}.rendered_smpl_visible_mask.jpg'.format(int(frame_id)))
    else:
        mask_path = os.path.join(INTERCAP_DIR, '_rendered_mask', sub_id, obj_id, 'Seg_{}'.format(seg_id), 'Frames_Cam{}'.format(cam_id), '{:05d}.rendered_smpl_mask.jpg'.format(int(frame_id)))
    return mask_path


def get_object_mask_path(img_id, visiable=True):
    sub_id, obj_id, seg_id, cam_id, frame_id = img_id.split('_')
    if visiable:
        mask_path = os.path.join(INTERCAP_DIR, '_rendered_mask', sub_id, obj_id, 'Seg_{}'.format(seg_id), 'Frames_Cam{}'.format(cam_id), '{:05d}.rendered_obj_visible_mask.jpg'.format(int(frame_id)))
    else:
        mask_path = os.path.join(INTERCAP_DIR, '_rendered_mask', sub_id, obj_id, 'Seg_{}'.format(seg_id), 'Frames_Cam{}'.format(cam_id), '{:05d}.rendered_obj_mask.jpg'.format(int(frame_id)))
    return mask_path


def get_gt_mesh(img_id):
    sub_id, obj_id, seg_id, cam_id, frame_id = img_id.split('_')
    smpl_gt_mesh_path = os.path.join(INTERCAP_DIR, 'Res', sub_id, obj_id, 'Seg_{}'.format(seg_id), 'Mesh', '{:05d}_second.ply'.format(int(frame_id)))
    object_gt_mesh_path = os.path.join(INTERCAP_DIR, 'Res', sub_id, obj_id, 'Seg_{}'.format(seg_id), 'Mesh', '{:05d}_second_obj.ply'.format(int(frame_id)))

    cam_id = int(cam_id)
    if cam_id == 1:
        cam_params = load_json('data/intercap/calibration_third/Color.json')
    else:
        cam_params = load_json('data/intercap/calibration_third/Color_{}.json'.format(cam_id))
    rotmat = R.from_rotvec(np.array(cam_params['R'])).as_matrix()
    t = np.array(cam_params['T'])
    gt_smpl_mesh = trimesh.load(smpl_gt_mesh_path, process=False)
    gt_smpl_mesh.vertices = np.matmul(gt_smpl_mesh.vertices, rotmat.T) + t

    gt_object_mesh = trimesh.load(object_gt_mesh_path, process=False)
    gt_object_mesh.vertices = np.matmul(gt_object_mesh.vertices, rotmat.T) + t

    return gt_smpl_mesh, gt_object_mesh
