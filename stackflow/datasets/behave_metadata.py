import os
import json
import trimesh
import numpy as np
import torch

BEHAVE_DIR = '/public/home/huochf/datasets/BEHAVE/'
FAKE_IMAGE_DIR = '/public/home/huochf/datasets/BEHAVE/rendered_images/'

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
    # 'person': 20,
}

OBJECT_IDX2NAME = {v: k for k, v in OBJECT_NAME2IDX.items()}

OBJECT_NAME_MAP = {'basketball': 'sports ball', 
                    'chairblack': 'chair',
                    'chairwood': 'chair',
                    'yogaball': 'sports ball',
                    'chairblack': 'chair',}

IMG_HEIGHT = 1536
IMG_WIDTH = 2048
MAX_OBJECT_VERT_NUM = 1700

SMPLH_OPENPOSE_INDICES = [52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                          8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                          60, 61, 62]

OBJECT_MESH_VERTEX_NUM = {
    "backpack": 548,
    "basketball": 508,
    "boxlarge": 524,
    "boxlong": 526,
    "boxmedium": 509,
    "boxsmall": 517,
    "boxtiny": 506,
    "chairblack": 1609,
    "chairwood": 1609,
    "keyboard": 502,
    "monitor": 537,
    "plasticcontainer": 562,
    "stool": 532,
    "suitcase": 520,
    "tablesmall": 507,
    "tablesquare": 1046,
    "toolbox": 499,
    "trashbin": 547,
    "yogaball": 534,
    "yogamat": 525,
}

def get_img_path_from_id(img_id, root=None, data_type='real'):
    if data_type == 'real':
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        if root is None:
            root = BEHAVE_DIR
        if inter_type != 'default':
            image_path = os.path.join(root, 'sequences', 'Date0{}_Sub0{}_{}_{}'.format(day_id, sub_id, obj_name, inter_type), 't0{}.000'.format(frame_id), 'k{}.color.jpg'.format(cam_id))
        else:
            image_path = os.path.join(root, 'sequences', 'Date0{}_Sub0{}_{}'.format(day_id, sub_id, obj_name), 't0{}.000'.format(frame_id), 'k{}.color.jpg'.format(cam_id))
    else:
        day_id, sub_id, obj_name, inter_type, frame_id, avatar_id, avatar_cloth, cam_id = img_id.split('_')
        avatar_id = avatar_id + '_' + avatar_cloth
        if root is None:
            root = BEHAVE_DIR
        if inter_type != 'default':
            image_path = os.path.join(root, 'rendered_images', 'Date0{}_Sub0{}_{}_{}'.format(day_id, sub_id, obj_name, inter_type), 't0{}.000'.format(frame_id), avatar_id, 'k{:02d}_hoi_color.jpg'.format(int(cam_id)))
        else:
            image_path = os.path.join(root, 'rendered_images', 'Date0{}_Sub0{}_{}'.format(day_id, sub_id, obj_name), 't0{}.000'.format(frame_id), avatar_id, 'k{:02d}_hoi_color.jpg'.format(int(cam_id)))
    return image_path


def get_gt_mesh(img_id):
    day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
    if inter_type == 'default':
        smpl_gt_mesh_path = os.path.join(BEHAVE_DIR, 'sequences/Date0{}_Sub0{}_{}/t0{}.000/person/fit02/person_fit.ply'.format(
                day_id, sub_id, obj_name, frame_id))
    else:
        smpl_gt_mesh_path = os.path.join(BEHAVE_DIR, 'sequences/Date0{}_Sub0{}_{}_{}/t0{}.000/person/fit02/person_fit.ply'.format(
                day_id, sub_id, obj_name, inter_type, frame_id))

    if inter_type == 'default':
        object_gt_mesh_path = os.path.join(BEHAVE_DIR, 'sequences/Date0{}_Sub0{}_{}/t0{}.000/{}/fit01/{}_fit.ply'.format(
                day_id, sub_id, obj_name, frame_id, obj_name, obj_name))
    else:
        object_gt_mesh_path = os.path.join(BEHAVE_DIR, 'sequences/Date0{}_Sub0{}_{}_{}/t0{}.000/{}/fit01/{}_fit.ply'.format(
                day_id, sub_id, obj_name, inter_type, frame_id, obj_name, obj_name))
    if not os.path.exists(object_gt_mesh_path):
        _obj_name = OBJECT_NAME_MAP[obj_name]
        if inter_type == 'default':
            object_gt_mesh_path = os.path.join(BEHAVE_DIR, 'sequences/Date0{}_Sub0{}_{}/t0{}.000/{}/fit01/{}_fit.ply'.format(
                    day_id, sub_id, obj_name, frame_id, _obj_name, _obj_name))
        else:
            object_gt_mesh_path = os.path.join(BEHAVE_DIR, 'sequences/Date0{}_Sub0{}_{}_{}/t0{}.000/{}/fit01/{}_fit.ply'.format(
                    day_id, sub_id, obj_name, inter_type, frame_id, _obj_name, _obj_name))

    cam_params_path = os.path.join(BEHAVE_DIR, 'calibs/Date0{}/config/{}/config.json'.format(day_id, cam_id))
    with open(cam_params_path, 'r') as f:
        cam_params = json.load(f)
    R = np.array(cam_params['rotation']).reshape((3, 3))
    t = np.array(cam_params['translation'])
    gt_smpl_mesh = trimesh.load(smpl_gt_mesh_path, process=False)
    gt_smpl_mesh.vertices = np.matmul((gt_smpl_mesh.vertices - t), R)

    gt_object_mesh = trimesh.load(object_gt_mesh_path, process=False)
    gt_object_mesh.vertices = np.matmul((gt_object_mesh.vertices - t), R)

    return gt_smpl_mesh, gt_object_mesh


def load_object_mesh_templates():
    templates = {}
    for object_name in OBJECT_NAME2IDX.keys():
        object_mesh = os.path.join(BEHAVE_DIR, 'objects', object_name, '{}_f1000.ply'.format(object_name))
        if not os.path.exists(object_mesh):
            object_mesh = os.path.join(BEHAVE_DIR, 'objects', object_name, '{}_f2000.ply'.format(object_name))
        if not os.path.exists(object_mesh):
            object_mesh = os.path.join(BEHAVE_DIR, 'objects', object_name, '{}_f2500.ply'.format(object_name))
        if not os.path.exists(object_mesh):
            object_mesh = os.path.join(BEHAVE_DIR, 'objects', object_name, '{}_closed_f1000.ply'.format(object_name))

        object_mesh = trimesh.load(object_mesh, process=False)
        verts = torch.tensor(object_mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(object_mesh.faces, dtype=torch.int64)
        verts = verts - verts.mean(0, keepdim=True)

        templates[object_name] = (verts, faces)

    return templates


def load_cam_intrinsics():
    cam_intrinsics_params = []
    for i in range(4):
        with open(os.path.join(BEHAVE_DIR, 'calibs', 'intrinsics', str(i), 'calibration.json'), 'r') as f:
            params = json.load(f)
        cx, cy = params['color']['cx'], params['color']['cy']
        fx, fy = params['color']['fx'], params['color']['fy']
        cam_intrinsics_params.append([cx, cy, fx, fy])
    return cam_intrinsics_params


def load_cam_RT_matrix():
    cam_RT = {}
    for day_id in range(7):
        cam_RT[str(day_id + 1)] = []
        for cam_id in range(4):
            with open(os.path.join(BEHAVE_DIR, 'calibs', 'Date0{}'.format(day_id + 1), 'config', str(cam_id), 'config.json'), 'r') as f:
                params = json.load(f)
            cam_RT[str(day_id + 1)].append([np.array(params['rotation']).reshape((3, 3)), np.array(params['translation'])])
    return cam_RT
