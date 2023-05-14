import os
import sys
import json
import argparse
import numpy as np
import random
import torch
import cv2
from scipy.spatial.transform import Rotation as R
from smplx import SMPLH
from sklearn.decomposition import PCA

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    look_at_view_transform,
    PointsRasterizer,
    PointsRenderer,
    PointsRasterizationSettings,
    NormWeightedCompositor,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    MeshRenderer,
    MeshRasterizer,
)

from prohmr.utils.geometry import perspective_projection

from stackflow.datasets.behave_hoi_template_dataset import BEHAVEHOITemplateDataset
from stackflow.datasets.data_utils import (
    extract_bbox_from_mask, 
    load_mask, 
    load_pickle, 
    save_pickle,
    load_json,
    save_json
)
from stackflow.datasets.behave_metadata import (
    BEHAVE_DIR,
    FAKE_IMAGE_DIR,
    OBJECT_NAME_MAP, 
    OBJECT_MESH_VERTEX_NUM,
    IMG_HEIGHT,
    IMG_WIDTH,
    OBJECT_NAME2IDX, 
    load_cam_RT_matrix,
    load_cam_intrinsics,
    load_object_mesh_templates
)


def preprocess_real(split='train'):
    smpl = SMPLH(model_path='data/smplh', gender='male')
    data_list = []

    cam_RT_matrix = load_cam_RT_matrix()
    cam_intrinsics_params = load_cam_intrinsics()
    mesh_templates = load_object_mesh_templates()

    with open(os.path.join(BEHAVE_DIR, 'split.json')) as f:
        sequence_list = json.load(f)
    sequence_list = sequence_list[split]

    sequence_dir = os.path.join(BEHAVE_DIR, 'sequences')
    for sequence_name in sequence_list:
        try:
            day_name, sub_name, obj_name, inter_type = sequence_name.split('_')
        except:
            day_name, sub_name, obj_name = sequence_name.split('_')
            inter_type = 'default'
        day_id, sub_id = day_name[5:], sub_name[4:]

        for frame_name in os.listdir(os.path.join(sequence_dir, sequence_name)):
            if frame_name == 'info.json':
                continue
            frame_id = frame_name[2:5]

            frame_dir = os.path.join(sequence_dir, sequence_name, frame_name)
            for cam_id in range(4):
                img_id = '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, str(cam_id)])

                person_mask = load_mask(os.path.join(frame_dir, 'k{}.person_mask.jpg'.format(cam_id)))
                object_mask = load_mask(os.path.join(frame_dir, 'k{}.obj_rend_mask.jpg'.format(cam_id)))

                person_bb_xyxy = extract_bbox_from_mask(person_mask)
                object_bb_xyxy = extract_bbox_from_mask(object_mask)

                object_param_path = os.path.join(frame_dir, obj_name, 'fit01', '{}_fit.pkl'.format(obj_name))
                if not os.path.exists(object_param_path):
                    _obj_name = OBJECT_NAME_MAP[obj_name]
                    object_param_path = os.path.join(frame_dir, _obj_name, 'fit01', '{}_fit.pkl'.format(_obj_name))
                object_params = load_pickle(object_param_path)

                cam_R, cam_T = cam_RT_matrix[day_id][cam_id]
                load_pickle(object_param_path)
                object_trans = object_params['trans']
                object_trans = np.matmul(cam_R.transpose(), object_trans - cam_T)
                object_pose = object_params['angle']
                object_rotmat = R.from_rotvec(object_pose).as_matrix()
                object_rotmat = np.matmul(cam_R.transpose(), object_rotmat)

                smplh_params = load_pickle(os.path.join(frame_dir, 'person', 'fit02', 'person_fit.pkl'))
                smplh_trans = smplh_params['trans']
                smplh_trans = np.matmul(cam_R.transpose(), smplh_trans - cam_T)
                smplh_betas = smplh_params['betas']
                smplh_pose = smplh_params['pose']

                global_pose = smplh_pose[:3]
                body_pose = smplh_pose[3:72]
                global_pose_rotmat = R.from_rotvec(global_pose).as_matrix()
                global_pose_rotmat = np.matmul(cam_R.transpose(), global_pose_rotmat)
                global_pose = R.from_matrix(global_pose_rotmat).as_rotvec()
                smplh_pose[:3] = global_pose
                pose_rotmat = R.from_rotvec(smplh_pose.reshape((-1, 3))).as_matrix()

                cx, cy, fx, fy = cam_intrinsics_params[cam_id]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

                smpl_out = smpl(betas=torch.tensor(smplh_betas, dtype=torch.float32).reshape(1, 10),
                                body_pose=torch.tensor(smplh_pose[3:66], dtype=torch.float32).reshape(1, 63),
                                global_orient=torch.tensor(smplh_pose[:3], dtype=torch.float32).reshape(1, 3),
                                transl=torch.tensor(smplh_trans, dtype=torch.float32).reshape(1, 3))
                joint_3d = smpl_out.joints
                joint_2d = perspective_projection(joint_3d,
                                                  translation=torch.zeros((1, 3)),
                                                  focal_length=torch.tensor([fx, fy]).unsqueeze(0),
                                                  camera_center=torch.tensor([cx, cy]).unsqueeze(0))
                joint_3d = joint_3d.detach().numpy().reshape(73, 3)
                joint_2d = joint_2d.detach().numpy().reshape(73, 2)

                coor_map_dir = os.path.join(BEHAVE_DIR, 'object_coor_maps_real', sequence_name, 't0{}.000'.format(frame_id), )
                if not os.path.exists(coor_map_dir):
                    os.makedirs(coor_map_dir)
                coor_map_path = os.path.join(coor_map_dir, 'k{}.obj_coor.pkl'.format(cam_id))
                mesh_verts, mesh_faces = mesh_templates[obj_name]
                rendered_obj_mask, _ = generate_object_coor_map(mesh_verts, mesh_faces, object_trans, object_rotmat, cx, cy, fx, fy, coor_map_path)
                if rendered_obj_mask is None:
                    visible_ratio = 0
                else:
                    visible_ratio = np.sum(object_mask != 0) / (np.sum(rendered_obj_mask != 0) + 1e-8)
                    visible_ratio = min(visible_ratio, 1.0)

                item = {}
                item['img_id'] = img_id
                item['data_type'] = 'real'
                item['person_bb_xyxy'] = person_bb_xyxy.astype(np.float32) # (4, )
                item['object_bb_xyxy'] = object_bb_xyxy.astype(np.float32) # (4, )
                item['smplh_pose'] = smplh_pose.astype(np.float32) # (156, )
                item['smplh_global_pose_rotmat'] = global_pose_rotmat.astype(np.float32) # (3, 3)
                item['smplh_pose_rotmat'] = pose_rotmat.astype(np.float32) # (52, 3, 3)
                item['smplh_betas'] = smplh_betas.astype(np.float32) # (10, )
                item['smplh_trans'] = smplh_trans.astype(np.float32) # (3, )
                item['smplh_joints_3d'] = joint_3d.astype(np.float32) # (73, 3)
                item['smplh_joints_2d'] = joint_2d.astype(np.float32) # (73, 2)
                item['object_trans'] = object_trans.astype(np.float32) # (3, )
                item['object_rotmat'] = object_rotmat.astype(np.float32) # (3, 3)
                item['object_label'] = OBJECT_NAME2IDX[obj_name]
                item['object_visible_ratio'] = visible_ratio
                item['cam_intrinsics'] = K.astype(np.float32) # (3, 3, )

                data_list.append(item)
                print('{}, Done!'.format(img_id))
                sys.stdout.flush()
        #         break
        #     break
        # break

    if not os.path.exists('data/behave'):
        os.makedirs('data/behave')
    save_pickle(data_list, 'data/behave/behave_real_{}_data_list.pkl'.format(split))


def preprocess_fake():
    data_list = []
    mesh_templates = load_object_mesh_templates()
    smplh = SMPLH(model_path='data/smplh', gender='male')

    fake_img_dir = FAKE_IMAGE_DIR
    sequence_list = os.listdir(fake_img_dir)
    for sequence_name in sequence_list:
        try:
            day_name, sub_name, obj_name, inter_type = sequence_name.split('_')
        except:
            day_name, sub_name, obj_name = sequence_name.split('_')
            inter_type = 'default'
        day_id, sub_id = day_name[5:], sub_name[4:]
        for frame_name in os.listdir(os.path.join(fake_img_dir, sequence_name)):
            frame_id = frame_name[2:5]
            frame_dir = os.path.join(fake_img_dir, sequence_name, frame_name)
            for avatar_id in os.listdir(os.path.join(frame_dir))[:1]:
                for cam_id in range(12):
                    img_id = '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, avatar_id, str(cam_id)])

                    person_mask = load_mask(os.path.join(frame_dir, avatar_id, 'k{:02d}_person_mask.jpg'.format(cam_id)))
                    object_mask = load_mask(os.path.join(frame_dir, avatar_id, 'k{:02d}_object_mask.jpg'.format(cam_id)))

                    person_bb_xyxy = extract_bbox_from_mask(person_mask)
                    object_bb_xyxy = extract_bbox_from_mask(object_mask)

                    params = load_json(os.path.join(frame_dir, avatar_id, 'k{:02d}_params.json'.format(cam_id)))
                    smplh_betas = np.array(params['smplh_params']['betas'], dtype=np.float32).reshape(10, )
                    smplh_pose = np.array(params['smplh_params']['pose'], dtype=np.float32).reshape(72, )
                    smplh_trans = np.array(params['smplh_params']['transl'], dtype=np.float32).reshape(3, )
                    global_pose_rotmat = R.from_rotvec(smplh_pose[:3]).as_matrix()
                    pose_rotmat = R.from_rotvec(smplh_pose.reshape(24, 3)).as_matrix()

                    object_rotmat = np.array(params['object_RT']['R'], dtype=np.float32).reshape((3, 3))
                    object_trans = np.array(params['object_RT']['T'], dtype=np.float32).reshape(3, )

                    cx, cy, fx, fy = 1018.952, 779.486, 979.7844, 979.840
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    smpl_out = smplh(betas=torch.tensor(smplh_betas, dtype=torch.float32).reshape(1, 10), 
                                     body_pose=torch.tensor(smplh_pose[3:66], dtype=torch.float32).reshape(1, 63), 
                                     global_orient=torch.tensor(smplh_pose[:3], dtype=torch.float32).reshape(1, 3), 
                                     transl=torch.tensor(smplh_trans, dtype=torch.float32).reshape(1, 3))
                    joint_3d = smpl_out.joints
                    joint_2d = perspective_projection(joint_3d,
                                                      translation=torch.zeros((1, 3)),
                                                      focal_length=torch.tensor([fx, fy]).unsqueeze(0),
                                                      camera_center=torch.tensor([cx, cy]).unsqueeze(0))
                    joint_3d = joint_3d.detach().numpy().reshape(73, 3)
                    joint_2d = joint_2d.detach().numpy().reshape(73, 2)

                    coor_map_dir = os.path.join(BEHAVE_DIR, 'object_coor_maps_fake', sequence_name, 't0{}.000'.format(frame_id), avatar_id)
                    if not os.path.exists(coor_map_dir):
                        os.makedirs(coor_map_dir)
                    coor_map_path = os.path.join(coor_map_dir, 'k{:02d}.obj_coor.pkl'.format(cam_id))
                    mesh_verts, mesh_faces = mesh_templates[obj_name]
                    object_mask, object_depth = generate_object_coor_map(mesh_verts, mesh_faces, object_trans, object_rotmat, cx, cy, fx, fy, coor_map_path)

                    if object_depth is None:
                        visible_ratio = 0.
                    else:
                        smpl_depth = get_smpl_depth(smpl_out.vertices, smplh.faces, cx, cy, fx, fy)
                        union_mask = (person_mask != 0) & (object_mask != 0)
                        visible_mask = (object_mask != 0) & (person_mask == 0) | (object_depth < smpl_depth) & union_mask
                        visible_ratio = np.sum(visible_mask == 1) / np.sum(object_mask != 0)
                        visible_ratio = min(visible_ratio, 1.0)

                    item = {}
                    item['img_id'] = img_id
                    item['data_type'] = 'fake'
                    item['person_bb_xyxy'] = person_bb_xyxy.astype(np.float32) # (4, )
                    item['object_bb_xyxy'] = object_bb_xyxy.astype(np.float32) # (4, )
                    item['smplh_pose'] = smplh_pose.astype(np.float32) # (72, )
                    item['smplh_global_pose_rotmat'] = global_pose_rotmat.astype(np.float32) # (3, 3)
                    item['smplh_pose_rotmat'] = pose_rotmat.astype(np.float32) # (24, 3, 3)
                    item['smplh_betas'] = smplh_betas.astype(np.float32) # (10, )
                    item['smplh_trans'] = smplh_trans.astype(np.float32) # (3, )
                    item['smplh_joints_3d'] = joint_3d.astype(np.float32) # (73, 3)
                    item['smplh_joints_2d'] = joint_2d.astype(np.float32) # (73, 2)
                    item['object_trans'] = object_trans.astype(np.float32) # (3, )
                    item['object_rotmat'] = object_rotmat.astype(np.float32) # (3, 3)
                    item['object_label'] = OBJECT_NAME2IDX[obj_name]
                    item['object_visible_ratio'] = visible_ratio
                    item['cam_intrinsics'] = K.astype(np.float32) # (3, 3, )

                    data_list.append(item)
                    print('{}, Done!'.format(img_id))
                    sys.stdout.flush()
        #             break
        #         break
        #     break
        # break

    if not os.path.exists('data/behave'):
        os.makedirs('data/behave')
    save_pickle(data_list, 'data/behave/behave_fake_data_list.pkl')


def post_process():
    fake_data_list = load_pickle('data/behave/behave_fake_data_list.pkl')
    new_fake_data_list = []
    for item in fake_data_list:
        item['data_type'] = 'fake'
        new_fake_data_list.append(item)
        print(item['img_id'] + ' Done !')
    save_pickle(new_fake_data_list, 'data/behave/_behave_fake_data_list.pkl')


def split_fake_dataset():
    fake_data_list = load_pickle('data/behave/behave_fake_data_list.pkl')
    split_6 = []
    split_4 = []
    split_3 = []
    split_2 = []
    split_1 = []
    split_0 = []
    for item in fake_data_list:
        day_id, sub_id, obj_name, inter_type, frame_id, avatar_id, _, cam_id = item['img_id'].split('_')

        if (int(cam_id) + 1) % 6 == 0:
            split_6.append(item)
        if (int(cam_id) + 1) % 4 == 0:
            split_4.append(item)
        if (int(cam_id) + 1) % 3 == 0:
            split_3.append(item)
        if (int(cam_id) + 1) % 2 == 0:
            split_2.append(item)
        if (int(cam_id) + 1) % 1 == 0:
            split_1.append(item)
    save_pickle(split_6, 'data/behave/_behave_fake_data_list_split_6.pkl')
    save_pickle(split_4, 'data/behave/_behave_fake_data_list_split_4.pkl')
    save_pickle(split_3, 'data/behave/_behave_fake_data_list_split_3.pkl')
    save_pickle(split_2, 'data/behave/_behave_fake_data_list_split_2.pkl')
    save_pickle(split_1, 'data/behave/_behave_fake_data_list_split_1.pkl')
    save_pickle(split_0, 'data/behave/_behave_fake_data_list_split_0.pkl')


def split_image_wise():
    # np.random.seed(7)
    # random.seed(7)
    # real_train_data_list = load_pickle('data/behave/behave_real_train_data_list.pkl')
    # real_test_data_list = load_pickle('data/behave/behave_real_test_data_list.pkl')
    # data_list = real_train_data_list + real_test_data_list
    # total_number = len(data_list)
    # random.shuffle(data_list)
    # trainset_size = int(0.8 * total_number)
    # datalist_train = data_list[:trainset_size]
    # datalist_test = data_list[trainset_size:]
    # print('Total size: {}, number of training samples: {}, number of test samples: {}'.format(total_number, len(datalist_train), len(datalist_test)))
    # save_pickle(datalist_train, 'data/behave/behave_real_train_data_list_image_wise_split.pkl')
    # save_pickle(datalist_test, 'data/behave/behave_real_test_data_list_image_wise_split.pkl')
    # image_id_train = [item['img_id'] for item in datalist_train]
    # image_id_test = [item['img_id'] for item in datalist_test]
    # save_json({'train': image_id_train, 'test': image_id_test}, 'data/behave/behave_image_wise_split.json')

    split = load_json('data/behave/behave_image_wise_split.json')
    split_out = {'train': [], 'test': []}
    for img_id in split['train']:
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        if inter_type != 'default':
            path = 'Date0{}_Sub0{}_{}_{}/t{:04d}.000/t{:04d}.000_k{}_scale.npz'.format(day_id, sub_id, obj_name, inter_type, int(frame_id), int(frame_id), cam_id)
        else:
            path = 'Date0{}_Sub0{}_{}/t{:04d}.000/t{:04d}.000_k{}_scale.npz'.format(day_id, sub_id, obj_name, int(frame_id), int(frame_id), cam_id)
        split_out['train'].append(path)
    for img_id in split['test']:
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        if inter_type != 'default':
            path = 'Date0{}_Sub0{}_{}_{}/t{:04d}.000/t{:04d}.000_k{}_scale.npz'.format(day_id, sub_id, obj_name, inter_type, int(frame_id), int(frame_id), cam_id)
        else:
            path = 'Date0{}_Sub0{}_{}/t{:04d}.000/t{:04d}.000_k{}_scale.npz'.format(day_id, sub_id, obj_name, int(frame_id), int(frame_id), cam_id)
        split_out['test'].append(path)
    save_pickle(split_out, 'data/behave/behave_image_wise_split.pkl')


def generate_object_coor_map(mesh_verts, mesh_faces, object_trans, object_rotmat, cx, cy, fx, fy, out_path):
    # resize with 0.5 ratio to save memory
    h, w = IMG_HEIGHT // 2, IMG_WIDTH // 2
    cx = cx / 2
    cy = cy / 2
    fx = fx / 2
    fy = fy / 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cam_R = torch.FloatTensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
    cam_T = torch.FloatTensor([[0, 0, 0,]]) 
    cameras = PerspectiveCameras(R=cam_R, T=cam_T,
                                 focal_length=[[fx, fy]], 
                                 principal_point=[[ cx,  cy]],
                                 image_size=[[h, w]],
                                 in_ndc=False,
                                 device=device)
    verts = torch.tensor(mesh_verts, dtype=torch.float32).to(device).reshape(-1, 3)
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

    if mask.sum() == 0:
        return None, None

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
    save_pickle(coor, out_path)
    print('Dump file: ' + out_path)
    sys.stdout.flush()
    # cv2.imwrite(out_path+'.jpg', 255 * cv2.resize(mask.detach().cpu().numpy(), (0, 0),  fx=2, fy=2, interpolation=cv2.INTER_NEAREST).astype(np.uint8))
    return (cv2.resize(mask.detach().cpu().numpy(), (0, 0),  fx=2, fy=2, interpolation=cv2.INTER_NEAREST), 
        cv2.resize(depth.detach().cpu().numpy(), (0, 0),  fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


def get_smpl_depth(smpl_v, smpl_f, cx, cy, fx, fy):

    h, w = 1536 // 2, 2048 // 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cam_R = torch.FloatTensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
    cam_T = torch.FloatTensor([[0, 0, 0]])            
    cameras = PerspectiveCameras(R=cam_R, T=cam_T,
                                 focal_length=[[fx / 2, fy / 2]], 
                                 principal_point=[[ cx / 2 ,  cy / 2]],
                                 image_size=[[h, w]],
                                 in_ndc=False,
                                 device=device)

    input_data = Meshes(verts=[smpl_v.cpu().reshape(-1, 3)], faces=[torch.tensor(smpl_f.astype(np.int64), dtype=torch.int64)], ).to(device)

    raster_settings = RasterizationSettings(image_size=[h, w], bin_size=0)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(input_data)
    depth = fragments.zbuf
    depth = depth.reshape(h, w)

    return cv2.resize(depth.detach().cpu().numpy(), (0, 0),  fx=2, fy=2, interpolation=cv2.INTER_NEAREST)


def sampl_smpl_object_anchors(smpl_anchor_num=[220, ], object_anchor_num=[16, ]):
    np.random.seed(7) # for reproducibility

    smpl_anchors, object_anchors = {}, {}
    smplh_pkl = load_pickle('data/smplh/SMPLH_MALE.pkl')
    weights = smplh_pkl['weights']
    parts_indices = np.argmax(weights[:, :22], axis=1)
    for num in smpl_anchor_num:
        num_per_parts = num // 22
        indices = []
        for i in range(22):
            part_anchor_indices = np.where(parts_indices == i)[0]
            part_anchors = part_anchor_indices[np.random.choice(len(part_anchor_indices), num_per_parts)]
            indices.append(part_anchors.tolist())
        smpl_anchors[num] = indices
    for num in object_anchor_num:
        object_anchors[num] = {}
        for obj_name, obj_vert_num in OBJECT_MESH_VERTEX_NUM.items():
            object_anchors[num][obj_name] = np.random.choice(obj_vert_num, num).tolist()

    save_json({'smpl': smpl_anchors, 'object': object_anchors}, 'data/behave/smplh_object_anchors_few.json')


def pca_analysis(object_anchor_indices, smpl_anchor_indices, lattent_feature_dim=32):
    pca_models = {}
    for obj_name in OBJECT_NAME2IDX.keys():
        dataset = BEHAVEHOITemplateDataset(datalist_file='data/behave/behave_real_train_data_list.pkl', object_name=obj_name)

        relative_distance = []
        for item in dataset:
            person_v = item['smplh_verts']
            object_v = item['object_verts']
            person_anchors = person_v[smpl_anchor_indices]
            object_anchors = object_v[object_anchor_indices[obj_name]]
            rel_dist = object_anchors.reshape((1, -1, 3)) - person_anchors.reshape((-1, 1, 3))
            rel_dist = rel_dist.reshape(-1)
            relative_distance.append(rel_dist)
        pca = PCA(n_components=lattent_feature_dim)
        relative_distance = np.stack(relative_distance, axis=0)
        new_X = pca.fit_transform(relative_distance)
        pca_models[obj_name] = {
            'mean': pca.mean_,
            'components': pca.components_,
            'smpl_anchor_indices': smpl_anchor_indices,
            'object_anchor_indices': object_anchor_indices[obj_name],
        }
    return pca_models


def extract_pca_models(pca_dim=32):
    smpl_object_anchors = load_json('data/behave/smplh_object_anchors_few.json')
    # for anchor_num in [8, 16, 32, 64]:
    for anchor_num in [4, 8, 16, 32,]:
        smpl_anchor_indices = np.array(smpl_object_anchors['smpl'][str(anchor_num * 22)]).reshape(-1, )
        object_anchor_indices = smpl_object_anchors['object'][str(anchor_num * 1)]
        pca_models = pca_analysis(object_anchor_indices, smpl_anchor_indices, pca_dim)
        out_dir = 'data/behave/behave_pca_models'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save_pickle(pca_models, os.path.join(out_dir, 'pca_models_n{}_d{}_few.pkl'.format(anchor_num, pca_dim)))
        print('save to ' + os.path.join(out_dir, 'pca_models_n{}_d{}_few.pkl'.format(anchor_num, pca_dim)))


def pca_analysis_iw(object_anchor_indices, smpl_anchor_indices, lattent_feature_dim=32):
    pca_models = {}
    for obj_name in OBJECT_NAME2IDX.keys():
        dataset = BEHAVEHOITemplateDataset(datalist_file='data/behave/behave_real_train_data_list_image_wise_split.pkl', object_name=obj_name)
        print(obj_name, len(dataset))

        relative_distance = []
        for item in dataset:
            person_v = item['smplh_verts']
            object_v = item['object_verts']
            person_anchors = person_v[smpl_anchor_indices]
            object_anchors = object_v[object_anchor_indices[obj_name]]
            rel_dist = object_anchors.reshape((1, -1, 3)) - person_anchors.reshape((-1, 1, 3))
            rel_dist = rel_dist.reshape(-1)
            relative_distance.append(rel_dist)
        pca = PCA(n_components=lattent_feature_dim)
        relative_distance = np.stack(relative_distance, axis=0)
        new_X = pca.fit_transform(relative_distance)
        pca_models[obj_name] = {
            'mean': pca.mean_,
            'components': pca.components_,
            'smpl_anchor_indices': smpl_anchor_indices,
            'object_anchor_indices': object_anchor_indices[obj_name],
        }
    return pca_models


def extract_pca_models_iw(pca_dim=32):
    smpl_object_anchors = load_json('data/behave/smplh_object_anchors.json')
    for anchor_num in [8, 16, 32, 64]:
        smpl_anchor_indices = np.array(smpl_object_anchors['smpl'][str(anchor_num * 22)]).reshape(-1, )
        object_anchor_indices = smpl_object_anchors['object'][str(anchor_num * 2)]
        pca_models = pca_analysis_iw(object_anchor_indices, smpl_anchor_indices, pca_dim)
        out_dir = 'data/behave/behave_pca_models'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save_pickle(pca_models, os.path.join(out_dir, 'pca_models_image_wise_n{}_d{}.pkl'.format(anchor_num, pca_dim)))
        print('save to ' + os.path.join(out_dir, 'pca_models_image_wise_n{}_d{}.pkl'.format(anchor_num, pca_dim)))


if __name__ == '__main__':
    preprocess_real('train')
    preprocess_real('test')
    preprocess_fake()
    # split_fake_dataset()
    # post_process()
    # split_image_wise()
    sampl_smpl_object_anchors(smpl_anchor_num=[176, 352, 704, 1408, 2200], object_anchor_num=[8, 16, 32, 64, 100, 128, 256])
    sampl_smpl_object_anchors(smpl_anchor_num=[88, 176, 352, 704], object_anchor_num=[4, 8, 16, 32])
    extract_pca_models()
    # extract_pca_models(pca_dim=64)
    # extract_pca_models(pca_dim=128)
    # extract_pca_models_iw()
