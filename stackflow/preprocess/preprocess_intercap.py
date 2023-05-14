import os
import sys
import random
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R
from smplx import SMPLX
from sklearn.decomposition import PCA
from tqdm import tqdm
from prohmr.utils.geometry import perspective_projection

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
from stackflow.datasets.intercap_hoi_template_dataset import InterCapHOITemplateDataset
from stackflow.datasets.data_utils import (
    load_pickle,
    save_pickle,
    extract_bbox_from_mask,
    load_mask,
    save_json,
    load_json
)
from stackflow.datasets.intercap_metadata import (
    INTERCAP_DIR,
    OBJECT_IDX2NAME,
    OBJECT_NAME2IDX,
    IMG_HEIGHT,
    IMG_WIDTH,
    OBJECT_MESH_VERTEX_NUM,
    get_image_path,
    load_cam_calibration,
    load_object_mesh_templates,
)


def preprocess():
    smpl = SMPLX(model_path='data/smplx', gender='male')
    data_list = []
    cam_calibration = load_cam_calibration()
    mesh_templates = load_object_mesh_templates()

    res_dir = os.path.join(INTERCAP_DIR, 'Res')
    image_dir = os.path.join(INTERCAP_DIR, 'RGBD_Images')
    for sub_id in sorted(os.listdir(image_dir)):
        for obj_id in sorted(os.listdir(os.path.join(image_dir, sub_id))):
            for seg_name in sorted(os.listdir(os.path.join(image_dir, sub_id, obj_id))):
                if 'Seg' not in seg_name:
                    continue
                object_annotations = load_pickle(os.path.join(res_dir, sub_id, obj_id, seg_name, 'res_obj.pkl'))
                person_annotations = load_pickle(os.path.join(res_dir, sub_id, obj_id, seg_name, 'res_1.pkl'))
                object_annotations_tuned = load_pickle(os.path.join(res_dir, sub_id, obj_id, seg_name, 'res_obj_tuned.pkl'))
                person_annotations_tuned = load_pickle(os.path.join(res_dir, sub_id, obj_id, seg_name, 'res_1_tuned.pkl'))
                for cam_name in sorted(os.listdir(os.path.join(image_dir, sub_id, obj_id, seg_name))):
                    for frame_name in sorted(os.listdir(os.path.join(image_dir, sub_id, obj_id, seg_name, cam_name, 'color', ))):

                        seg_id = seg_name[-1]
                        cam_id = cam_name[-1]
                        frame_id = frame_name.split('.')[0]
                        img_id = '_'.join([sub_id, obj_id, seg_id, cam_id, frame_id])

                        frame_index = int(frame_id)
                        if frame_index >= len(person_annotations) or frame_index >= len(person_annotations_tuned):
                            print('Skip {}!!!'.format(img_id)) # '04_07_0_*_50-263' '10_05_0_*_00120'
                            continue

                        calitration = cam_calibration[cam_id]
                        cam_R = np.array(calitration['R'])
                        cam_R = R.from_rotvec(cam_R).as_matrix()
                        cam_T = np.array(calitration['T'])
                        cx, cy = calitration['c']
                        fx, fy = calitration['f']
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

                        smplh_betas = person_annotations_tuned[frame_index]['betas'].reshape(10, )
                        smplh_global_pose = person_annotations_tuned[frame_index]['global_orient'].reshape(3, )
                        smplh_global_rotmat = R.from_rotvec(smplh_global_pose).as_matrix()
                        smplh_trans = person_annotations_tuned[frame_index]['transl'].reshape(3, )
                        smplh_body_pose = person_annotations_tuned[frame_index]['body_pose'].reshape((21, 3))

                        smplh_body_rotmat = R.from_rotvec(smplh_body_pose).as_matrix()
                        smplh_global_rotmat = np.matmul(cam_R, smplh_global_rotmat)
                        smplh_global_pose = R.from_matrix(smplh_global_rotmat).as_rotvec()
                        smplh_pose = np.concatenate([smplh_global_pose, smplh_body_pose.reshape(-1)], axis=0) # (66, )
                        smplh_pose_rotmat = np.concatenate([smplh_global_rotmat.reshape((1, 3, 3)), smplh_body_rotmat], axis=0)
                        smplh_trans = np.matmul(cam_R, smplh_trans) + cam_T

                        smpl_out = smpl(betas=torch.tensor(smplh_betas, dtype=torch.float32).reshape(1, 10),
                                        body_pose=torch.tensor(smplh_pose[3:66], dtype=torch.float32).reshape(1, 63),
                                        global_orient=0 * torch.tensor(smplh_pose[:3], dtype=torch.float32).reshape(1, 3),
                                        transl=0 * torch.tensor(smplh_trans, dtype=torch.float32).reshape(1, 3))
                        J_0 = smpl_out.joints[:, 0].detach().numpy().reshape(3, )
                        smplh_trans = smplh_trans + np.matmul(cam_R, J_0) - J_0

                        smpl_out = smpl(betas=torch.tensor(smplh_betas, dtype=torch.float32).reshape(1, 10),
                                        body_pose=torch.tensor(smplh_pose[3:66], dtype=torch.float32).reshape(1, 63),
                                        global_orient=torch.tensor(smplh_pose[:3], dtype=torch.float32).reshape(1, 3),
                                        transl=torch.tensor(smplh_trans, dtype=torch.float32).reshape(1, 3))
                        joint_3d = smpl_out.joints

                        joint_2d = perspective_projection(joint_3d,
                                                          translation=torch.zeros((1, 3)),
                                                          focal_length=torch.tensor([fx, fy]).unsqueeze(0),
                                                          camera_center=torch.tensor([cx, cy]).unsqueeze(0))
                        joint_3d = joint_3d.detach().numpy().reshape(127, 3)[:73]
                        joint_2d = joint_2d.detach().numpy().reshape(127, 2)[:73]
                        # print(joint_2d)

                        object_R = object_annotations_tuned[frame_index]['pose']
                        object_R = R.from_rotvec(object_R).as_matrix()
                        object_R = np.matmul(cam_R, object_R)
                        object_T = object_annotations_tuned[frame_index]['trans']
                        object_T = np.matmul(cam_R, object_T) + cam_T
                        object_mask = object_annotations[frame_index]['segs'][int(cam_id) - 1]

                        coor_map_dir = os.path.join(INTERCAP_DIR, 'object_coor_maps', sub_id, obj_id, seg_name, cam_name)
                        if not os.path.exists(coor_map_dir):
                            os.makedirs(coor_map_dir)
                        coor_map_path = os.path.join(coor_map_dir, '{:05d}.obj_coor.pkl'.format(frame_index))
                        mesh_verts, mesh_faces = mesh_templates[OBJECT_IDX2NAME[int(obj_id) - 1]]

                        mask_dir = os.path.join(INTERCAP_DIR, 'rendered_mask', sub_id, obj_id, seg_name, cam_name)
                        if True: # not os.path.exists(os.path.join(mask_dir, '{:05d}.rendered_obj_mask.jpg'.format(frame_index))):
                            rendered_obj_mask, rendered_obj_depth = generate_object_coor_map(mesh_verts, mesh_faces, object_T, object_R, cx, cy, fx, fy, coor_map_path)
                            rendered_smpl_mask, rendered_smpl_depth = render_smpl(smpl_out.vertices.detach(), smpl.faces, cx, cy, fx, fy,)

                            person_bb_xyxy = extract_bbox_from_mask(rendered_smpl_mask)
                            object_bb_xyxy = extract_bbox_from_mask(rendered_obj_mask)

                            if rendered_obj_mask is None:
                                visible_ratio = 0
                                object_visible_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH))
                                rendered_obj_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH))
                            else:
                                union_mask = (rendered_smpl_mask != 0) & (rendered_obj_mask != 0)
                                visible_mask = (rendered_obj_mask != 0) & (rendered_smpl_mask == 0) | (rendered_obj_depth < rendered_smpl_depth) & union_mask
                                visible_ratio = np.sum(visible_mask == 1) / np.sum(rendered_obj_mask != 0)
                                visible_ratio = min(visible_ratio, 1.0)
                                object_visible_mask = visible_mask
                                smpl_visible_mask = (rendered_smpl_mask != 0) & (rendered_obj_mask == 0) | (rendered_smpl_depth < rendered_obj_depth) & union_mask
                            mask_dir = os.path.join(INTERCAP_DIR, 'rendered_mask', sub_id, obj_id, seg_name, cam_name)
                            if not os.path.exists(mask_dir):
                                os.makedirs(mask_dir)
                            cv2.imwrite(os.path.join(mask_dir, '{:05d}.rendered_obj_mask.jpg'.format(frame_index)), (255 * rendered_obj_mask).astype(np.uint8))
                            cv2.imwrite(os.path.join(mask_dir, '{:05d}.rendered_obj_visible_mask.jpg'.format(frame_index)), (255 * object_visible_mask).astype(np.uint8))
                            cv2.imwrite(os.path.join(mask_dir, '{:05d}.rendered_smpl_mask.jpg'.format(frame_index)), (255 * rendered_smpl_mask).astype(np.uint8))
                            cv2.imwrite(os.path.join(mask_dir, '{:05d}.rendered_smpl_visible_mask.jpg'.format(frame_index)), (255 * smpl_visible_mask).astype(np.uint8))
                        else:
                            rendered_obj_mask = load_mask(os.path.join(mask_dir, '{:05d}.rendered_obj_mask.jpg'.format(frame_index)))
                            object_visible_mask = load_mask(os.path.join(mask_dir, '{:05d}.rendered_obj_visible_mask.jpg'.format(frame_index)))
                            rendered_smpl_mask = load_mask(os.path.join(mask_dir, '{:05d}.rendered_smpl_mask.jpg'.format(frame_index)))
                            smpl_visible_mask = load_mask(os.path.join(mask_dir, '{:05d}.rendered_smpl_visible_mask.jpg'.format(frame_index)))

                            person_bb_xyxy = extract_bbox_from_mask(rendered_smpl_mask)
                            object_bb_xyxy = extract_bbox_from_mask(rendered_obj_mask)

                            visible_ratio = np.sum(object_visible_mask != 0) / np.sum(rendered_obj_mask != 0)
                            visible_ratio = min(visible_ratio, 1.0)

                        item = {}
                        item['img_id'] = img_id
                        item['person_bb_xyxy'] = person_bb_xyxy.astype(np.float32) # (4, )
                        item['object_bb_xyxy'] = object_bb_xyxy.astype(np.float32) # (4, )
                        item['smplh_pose'] = smplh_pose.astype(np.float32) # (66, )
                        item['smplh_global_pose_rotmat'] = smplh_global_rotmat.astype(np.float32) # (3, 3)
                        item['smplh_pose_rotmat'] = smplh_pose_rotmat.astype(np.float32) # (22, 3, 3)
                        item['smplh_betas'] = smplh_betas.astype(np.float32) # (10, )
                        item['smplh_trans'] = smplh_trans.astype(np.float32) # (3, )
                        item['smplh_joints_3d'] = joint_3d.astype(np.float32) # (73, 3)
                        item['smplh_joints_2d'] = joint_2d.astype(np.float32) # (73, 2)
                        item['object_trans'] = object_T.astype(np.float32) # (3, )
                        item['object_rotmat'] = object_R.astype(np.float32) # (3, 3)
                        item['object_label'] = int(obj_id) - 1
                        item['object_visible_ratio'] = visible_ratio
                        item['cam_intrinsics'] = K.astype(np.float32) # (3, 3)

                        data_list.append(item)
                        print('{}, Done!'.format(img_id))
                        sys.stdout.flush()
                        # break
                    # break
                # break
            # break
        # break

    if not os.path.exists('data/intercap'):
        os.makedirs('data/intercap')
    save_pickle(data_list, 'data/intercap/intercap_data_list.pkl')


def preprocess_with_hand():
    smpl = SMPLX(model_path='data/smplx', gender='male')
    data_list = []
    cam_calibration = load_cam_calibration()
    mesh_templates = load_object_mesh_templates()

    res_dir = os.path.join(INTERCAP_DIR, 'Res')
    image_dir = os.path.join(INTERCAP_DIR, 'RGBD_Images')
    for sub_id in sorted(os.listdir(image_dir)):
        for obj_id in sorted(os.listdir(os.path.join(image_dir, sub_id))):
            for seg_name in sorted(os.listdir(os.path.join(image_dir, sub_id, obj_id))):
                if 'Seg' not in seg_name:
                    continue
                object_annotations = load_pickle(os.path.join(res_dir, sub_id, obj_id, seg_name, 'res_obj.pkl'))
                person_annotations = load_pickle(os.path.join(res_dir, sub_id, obj_id, seg_name, 'res_1.pkl'))
                object_annotations_tuned = load_pickle(os.path.join(res_dir, sub_id, obj_id, seg_name, 'res_obj_tuned.pkl'))
                person_annotations_tuned = load_pickle(os.path.join(res_dir, sub_id, obj_id, seg_name, 'res_1_tuned.pkl'))
                for cam_name in sorted(os.listdir(os.path.join(image_dir, sub_id, obj_id, seg_name))):
                    for frame_name in sorted(os.listdir(os.path.join(image_dir, sub_id, obj_id, seg_name, cam_name, 'color', ))):

                        seg_id = seg_name[-1]
                        cam_id = cam_name[-1]
                        frame_id = frame_name.split('.')[0]
                        img_id = '_'.join([sub_id, obj_id, seg_id, cam_id, frame_id])

                        frame_index = int(frame_id)
                        if frame_index >= len(person_annotations) or frame_index >= len(person_annotations_tuned):
                            print('Skip {}!!!'.format(img_id)) # '04_07_0_*_50-263' '10_05_0_*_00120'
                            continue

                        calitration = cam_calibration[cam_id]
                        cam_R = np.array(calitration['R'])
                        cam_R = R.from_rotvec(cam_R).as_matrix()
                        cam_T = np.array(calitration['T'])
                        cx, cy = calitration['c']
                        fx, fy = calitration['f']
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

                        smplh_betas = person_annotations_tuned[frame_index]['betas'].reshape(10, )
                        smplh_global_pose = person_annotations_tuned[frame_index]['global_orient'].reshape(3, )
                        smplh_global_rotmat = R.from_rotvec(smplh_global_pose).as_matrix()
                        smplh_trans = person_annotations_tuned[frame_index]['transl'].reshape(3, )
                        smplh_body_pose = person_annotations_tuned[frame_index]['body_pose'].reshape((21, 3))

                        smplh_body_rotmat = R.from_rotvec(smplh_body_pose).as_matrix()
                        smplh_global_rotmat = np.matmul(cam_R, smplh_global_rotmat)
                        smplh_global_pose = R.from_matrix(smplh_global_rotmat).as_rotvec()
                        smplh_pose = np.concatenate([smplh_global_pose, smplh_body_pose.reshape(-1)], axis=0) # (66, )
                        smplh_pose_rotmat = np.concatenate([smplh_global_rotmat.reshape((1, 3, 3)), smplh_body_rotmat], axis=0)
                        smplh_trans = np.matmul(cam_R, smplh_trans) + cam_T

                        object_R = object_annotations_tuned[frame_index]['pose']
                        object_R = R.from_rotvec(object_R).as_matrix()
                        object_R = np.matmul(cam_R, object_R)
                        object_T = object_annotations_tuned[frame_index]['trans']
                        object_T = np.matmul(cam_R, object_T) + cam_T

                        left_hand_pose = person_annotations_tuned[frame_index]['left_hand_pose'].reshape(12, )
                        right_hand_pose = person_annotations_tuned[frame_index]['right_hand_pose'].reshape(12, )

                        item = {}
                        item['img_id'] = img_id
                        item['smplh_pose'] = smplh_pose.astype(np.float32) # (66, )
                        item['smplh_global_pose_rotmat'] = smplh_global_rotmat.astype(np.float32) # (3, 3)
                        item['smplh_pose_rotmat'] = smplh_pose_rotmat.astype(np.float32) # (22, 3, 3)
                        item['left_hand_pose'] = left_hand_pose.astype(np.float32) # (12, )
                        item['right_hand_pose'] = right_hand_pose.astype(np.float32) # (12, )
                        item['smplh_betas'] = smplh_betas.astype(np.float32) # (10, )
                        item['smplh_trans'] = smplh_trans.astype(np.float32) # (3, )
                        item['object_trans'] = object_T.astype(np.float32) # (3, )
                        item['object_rotmat'] = object_R.astype(np.float32) # (3, 3)
                        item['object_label'] = int(obj_id) - 1
                        item['cam_intrinsics'] = K.astype(np.float32) # (3, 3)

                        data_list.append(item)
                        print('{}, Done!'.format(img_id))
                        sys.stdout.flush()
                        # break
                    # break
                # break
            # break
        # break

    if not os.path.exists('data/intercap'):
        os.makedirs('data/intercap')
    save_pickle(data_list, 'data/intercap/intercap_data_list_with_hand.pkl')


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


def render_smpl(smpl_v, smpl_f, cx, cy, fx, fy):

    h, w = IMG_HEIGHT // 2, IMG_WIDTH // 2
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
    mask = torch.zeros_like(depth)
    mask[depth != -1] = 1

    return (cv2.resize(mask.detach().cpu().numpy(), (0, 0),  fx=2, fy=2, interpolation=cv2.INTER_NEAREST), 
        cv2.resize(depth.detach().cpu().numpy(), (0, 0),  fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


def filter_dataset():
    data_list = load_pickle('data/intercap/intercap_data_list.pkl', )
    new_data_list = []
    for item in data_list:
        img_path = get_image_path(item['img_id'])
        if True in np.isnan(item['smplh_betas']):
            print('detect nan vlaues, ', item['img_id'])
        elif os.stat(img_path).st_size == 0:
            print('image: {} is empty'.format(get_image_path(item['img_id'])), item['img_id'])
        else:
            new_data_list.append(item)

    print(len(new_data_list), len(data_list))
    save_pickle(new_data_list, 'data/intercap/intercap_data_list_filtered.pkl')


def split_dataset():
    np.random.seed(7)
    random.seed(7)
    data_list = load_pickle('data/intercap/intercap_data_list_filtered.pkl', )
    total_number = len(data_list)
    random.shuffle(data_list)
    trainset_size = int(0.8 * total_number)
    datalist_train = data_list[:trainset_size]
    datalist_test = data_list[trainset_size:]
    print('Total size: {}, number of training samples: {}, number of test samples: {}'.format(total_number, len(datalist_train), len(datalist_test)))
    save_pickle(datalist_train, 'data/intercap/intercap_data_list_train_image_wise_split.pkl')
    save_pickle(datalist_test, 'data/intercap/intercap_data_list_test_image_wise_split.pkl')
    image_id_train = [item['img_id'] for item in datalist_train]
    image_id_test = [item['img_id'] for item in datalist_test]
    save_json({'train': image_id_train, 'test': image_id_test}, 'data/intercap/intercap_image_wise_split.json')

    datalist_train, datalist_test = [], []
    sequence_split = []
    total_sequence_count = 0
    sequence_candidates = []

    image_dir = os.path.join(INTERCAP_DIR, 'RGBD_Images')
    for sub_id in sorted(os.listdir(image_dir)):
        for obj_id in sorted(os.listdir(os.path.join(image_dir, sub_id))):
            seg_list = sorted(os.listdir(os.path.join(image_dir, sub_id, obj_id)))
            seg_list = [seg_name[-1] for seg_name in seg_list if 'Seg' in seg_name]
            sequence_candidates.append((sub_id, obj_id, seg_list[np.random.choice(len(seg_list))]))
            total_sequence_count += len(seg_list)

    random.shuffle(sequence_candidates)
    sequence_split = sequence_candidates[:int(0.2 * total_sequence_count)]

    for item in data_list:
        sub_id, obj_id, seg_id, cam_id, frame_id = item['img_id'].split('_')

        seq_id = (sub_id, obj_id, seg_id)
        if seq_id in sequence_split:
            datalist_test.append(item)
        else:
            datalist_train.append(item)
    print('Total size: {}, number of training samples: {}, number of test samples: {}'.format(total_number, len(datalist_train), len(datalist_test)))
    save_pickle(datalist_train, 'data/intercap/intercap_data_list_train_seq_wise_split.pkl')
    save_pickle(datalist_test, 'data/intercap/intercap_data_list_test_seq_wise_split.pkl')
    image_id_train = [item['img_id'] for item in datalist_train]
    image_id_test = [item['img_id'] for item in datalist_test]
    save_json({'train': image_id_train, 'test': image_id_test}, 'data/intercap/intercap_seq_wise_split.json')


def sample_smpl_object_anchors(smpl_anchor_num=[220, ], object_anchor_num=[16, ]):
    np.random.seed(7) # for reproducibility

    smpl_anchors, object_anchors = {}, {}
    smplh_pkl = load_pickle('data/smplx/SMPLX_MALE.pkl')
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

    save_json({'smpl': smpl_anchors, 'object': object_anchors}, 'data/intercap/smplx_object_anchors.json')


def pca_analysis(object_anchor_indices, smpl_anchor_indices, lattent_feature_dim=32):
    pca_models = {}
    for obj_name in OBJECT_NAME2IDX.keys():
        print(obj_name)
        dataset = InterCapHOITemplateDataset(datalist_file='data/intercap/intercap_data_list_train_seq_wise_split.pkl', object_name=obj_name)
        print(obj_name, len(dataset))
        relative_distance = []
        for item in dataset:
            person_v = item['smplh_verts']
            object_v = item['object_verts']
            person_anchors = person_v[smpl_anchor_indices]
            object_anchors = object_v[object_anchor_indices[obj_name]]
            rel_dist = object_anchors.reshape((1, -1, 3)) - person_anchors.reshape((-1, 1, 3))
            rel_dist = rel_dist.reshape(-1).numpy()
            if True not in np.isnan(rel_dist):
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
    smpl_object_anchors = load_json('data/intercap/smplx_object_anchors.json')
    for anchor_num in [8, 16, 32, 64]:
        smpl_anchor_indices = np.array(smpl_object_anchors['smpl'][str(anchor_num * 22)]).reshape(-1, )
        object_anchor_indices = smpl_object_anchors['object'][str(anchor_num * 2)]
        pca_models = pca_analysis(object_anchor_indices, smpl_anchor_indices, pca_dim)
        out_dir = 'data/intercap/intercap_pca_models'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save_pickle(pca_models, os.path.join(out_dir, 'pca_models_n{}_d{}_seq_wise.pkl'.format(anchor_num, pca_dim)))
        print('save to ' + os.path.join(out_dir, 'pca_models_n{}_d{}_seq_wise.pkl'.format(anchor_num, pca_dim)))


def sample_hand_object_anchors(hand_anchor_num=[128, 256, 512], object_anchor_num=[32, 64, 128],):
    np.random.seed(7) # for reproducibility
    hands_anchors, object_anchors = {}, {}

    smplx_pkl = load_pickle('data/smplx/SMPLX_MALE.pkl')
    v_labels = np.argmax(smplx_pkl['weights'], axis=1)
    part_joint_mapping = {
        'head': [12, 15, 22, 23, 24, 55, ],
        'torso': [0, 3, 6, 9, 13, 14],
        
        'left_foot': [7, 10, ],
        'upper_left_leg': [4, ],
        'left_leg': [1, ],
        'left_forearm': [20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        'left_midarm': [18, ],
        'left_upperarm': [16, ],
        
        'right_foot': [8, 11, ],
        'upper_right_leg': [5, ],
        'right_leg': [2, ],
        'right_forearm': [21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'right_midarm': [19, ],
        'right_upperarm': [17, ],
    }
    for num in hand_anchor_num:
        left_hand_anchors, right_hand_anchors = [], []
        print(num, num // 16)
        for i in range(16):
            left_hand_indices = np.where(v_labels == part_joint_mapping['left_forearm'][i])[0]
            left_hand_anchors.extend(left_hand_indices[np.random.choice(len(left_hand_indices), num // 16)].tolist())

        for i in range(16):
            right_hand_indices = np.where(v_labels == part_joint_mapping['right_forearm'][i])[0]
            right_hand_anchors.extend(right_hand_indices[np.random.choice(len(right_hand_indices), num // 16)].tolist())

        hands_anchors[num] = {
            'left': left_hand_anchors,
            'right': right_hand_anchors,
        }

    for num in object_anchor_num:
        object_anchors[num] = {}
        for obj_name, obj_vert_num in OBJECT_MESH_VERTEX_NUM.items():
            object_anchors[num][obj_name] = np.random.choice(obj_vert_num, num).tolist()
    save_json({'hand': hands_anchors, 'object': object_anchors}, 'data/intercap/smplx_hand_object_anchors.json')


def pca_hand_analysis(object_anchor_indices, hand_anchor_indices, lattent_feature_dim=32):
    pca_models = {}
    for obj_name in OBJECT_NAME2IDX.keys():
        print(obj_name)
        dataset = InterCapHOITemplateDataset(datalist_file='data/intercap/intercap_data_list_train_seq_wise_split.pkl', object_name=obj_name)
        print(obj_name, len(dataset))
        left_relative_distance, right_relative_distance = [], []
        for item in tqdm(dataset):
            person_v = item['smplh_verts']
            object_v = item['object_verts']
            if item['left_in_contact']:
                hand_anchors = person_v[hand_anchor_indices['left']]
                object_anchors = object_v[object_anchor_indices[obj_name]]

                rel_dist = object_anchors.reshape((1, -1, 3)) - hand_anchors.reshape((-1, 1, 3))
                rel_dist = rel_dist.reshape(-1).numpy()
                if True not in np.isnan(rel_dist):
                    left_relative_distance.append(rel_dist)

            if item['right_in_contact']:
                hand_anchors = person_v[hand_anchor_indices['right']]
                object_anchors = object_v[object_anchor_indices[obj_name]]

                rel_dist = object_anchors.reshape((1, -1, 3)) - hand_anchors.reshape((-1, 1, 3))
                rel_dist = rel_dist.reshape(-1).numpy()
                if True not in np.isnan(rel_dist):
                    right_relative_distance.append(rel_dist)

        try:
            pca = PCA(n_components=lattent_feature_dim)
            left_relative_distance = np.stack(left_relative_distance, axis=0)
            print('left: ', left_relative_distance.shape)
            new_X = pca.fit_transform(left_relative_distance)
            pca_models[obj_name] = {}
            pca_models[obj_name]['left'] = {
                'mean': pca.mean_,
                'components': pca.components_,
                'smpl_anchor_indices': hand_anchor_indices['left'],
                'object_anchor_indices': object_anchor_indices[obj_name],
            }
            
            pca = PCA(n_components=lattent_feature_dim)
            right_relative_distance = np.stack(right_relative_distance, axis=0)
            print('right: ', right_relative_distance.shape)
            new_X = pca.fit_transform(right_relative_distance)
            pca_models[obj_name]['right'] = {
                'mean': pca.mean_,
                'components': pca.components_,
                'smpl_anchor_indices': hand_anchor_indices['right'],
                'object_anchor_indices': object_anchor_indices[obj_name],
            }
        except:
            pca_models[obj_name] = {}

    return pca_models


def extract_hand_pca_models(pca_dim=32):
    hand_object_anchors = load_json('data/intercap/smplx_hand_object_anchors.json')
    for anchor_num in [32, 64, 128]:
        hand_anchors = hand_object_anchors['hand'][str(anchor_num * 4)]
        object_anchors = hand_object_anchors['object'][str(anchor_num)]
        pca_models = pca_hand_analysis(object_anchors, hand_anchors, pca_dim)
        out_dir = 'data/intercap/intercap_pca_models'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save_pickle(pca_models, os.path.join(out_dir, 'hand_pca_models_n{}_d{}_seq_wise.pkl'.format(anchor_num, pca_dim)))
        print('save to ' + os.path.join(out_dir, 'hand_pca_models_n{}_d{}_seq_wise.pkl'.format(anchor_num, pca_dim)))


if __name__ == '__main__':
    preprocess()
    filter_dataset()
    split_dataset()
    sample_smpl_object_anchors(smpl_anchor_num=[176, 352, 704, 1408, 2200], object_anchor_num=[8, 16, 32, 64, 100, 128, 256])
    extract_pca_models()
    # sample_hand_object_anchors()
    # preprocess_with_hand()
    # extract_hand_pca_models()
