import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
from tqdm import tqdm
import cv2
import argparse
import random
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from smplx import SMPLH, SMPLX, SMPLHLayer, SMPLXLayer

from pytorch3d.transforms import axis_angle_to_matrix

from stackflow.datasets.behave_metadata import BEHAVEMetaData
from stackflow.datasets.intercap_metadata import InterCapMetaData
from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from stackflow.datasets.utils import save_pickle, save_json, load_pickle
from stackflow.utils.camera import perspective_projection


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


def fit_smplh_male(smplh_betas, smplh_pose_rotmat, is_female):
    b = smplh_betas.shape[0]
    device = torch.device('cuda')
    smpl_male = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
    smpl_female = SMPLHLayer(model_path='data/models/smplh', gender='female').to(device)

    smplh_betas_male = nn.Parameter(smplh_betas.to(torch.float32).clone())
    smplh_betas = smplh_betas.to(torch.float32)
    smplh_pose_rotmat = smplh_pose_rotmat.to(torch.float32)
    global_orient = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3).repeat(b, 1, 1)
    transl = torch.zeros((b, 3), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([smplh_betas_male, ], lr=1e-1, betas=(0.9, 0.999))
    iterations = 1
    steps_per_iter = 100
    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        for i in loop:
            optimizer.zero_grad()

            smpl_female_out = smpl_female(betas=smplh_betas, body_pose=smplh_pose_rotmat[:, 1:22], global_orient=global_orient, transl=transl)
            smpl_male_out = smpl_male(betas=smplh_betas_male, body_pose=smplh_pose_rotmat[:, 1:22], global_orient=global_orient, transl=transl)

            smpl_female_v = smpl_female_out.vertices
            smpl_male_v = smpl_male_out.vertices
            smpl_female_joints = smpl_female_out.joints
            smpl_male_joints = smpl_male_out.joints
            loss = F.l1_loss(smpl_female_v, smpl_male_v, reduction='none').reshape(b, -1).mean(-1) + F.l1_loss(smpl_female_joints, smpl_male_joints, reduction='none').reshape(b, -1).mean(-1)
            loss = loss * is_female.reshape(b, )
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            l_str += ', {}: {:0.4f}'.format('loss', loss.item())
            loop.set_description(l_str)
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.1

    return smplh_betas_male.detach()


def fit_smplx_neutral(smplx_betas, smplx_pose_rotmat, is_female):
    b = smplx_betas.shape[0]
    device = torch.device('cuda')
    smpl_male = SMPLXLayer(model_path='data/models/smplx', gender='male').to(device)
    smpl_female = SMPLXLayer(model_path='data/models/smplx', gender='female').to(device)
    smpl_neutral = SMPLXLayer(model_path='data/models/smplx', gender='neutral').to(device)

    smplx_betas_neutral = nn.Parameter(smplx_betas.to(torch.float32).clone())
    smplx_betas = smplx_betas.to(torch.float32)
    smplx_pose_rotmat = smplx_pose_rotmat[:, :22].to(torch.float32)
    global_orient = torch.eye(3, dtype=torch.float32, device=device).repeat(b, 1, 1)
    transl = torch.zeros((b, 3), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([smplx_betas_neutral, ], lr=1e-1, betas=(0.9, 0.999))
    iterations = 1
    steps_per_iter = 100
    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        for i in loop:
            optimizer.zero_grad()

            smpl_neutral_out = smpl_neutral(betas=smplx_betas_neutral, body_pose=smplx_pose_rotmat[:, 1:22], global_orient=global_orient, transl=transl)
            smpl_neutral_v = smpl_neutral_out.vertices
            smpl_neutral_joints = smpl_neutral_out.joints

            smpl_out = smpl_female(betas=smplx_betas, body_pose=smplx_pose_rotmat[:, 1:22], global_orient=global_orient, transl=transl)
            smpl_v = smpl_out.vertices
            smpl_joints = smpl_out.joints
            loss = F.l1_loss(smpl_v, smpl_neutral_v, reduction='none').reshape(b, -1).mean(-1) + F.l1_loss(smpl_joints, smpl_neutral_joints, reduction='none').reshape(b, -1).mean(-1)
            loss_female = loss * is_female.reshape(b, 1, 1)

            smpl_out = smpl_male(betas=smplx_betas, body_pose=smplx_pose_rotmat[:, 1:22], global_orient=global_orient, transl=transl)
            smpl_v = smpl_out.vertices
            smpl_joints = smpl_out.joints
            loss = F.l1_loss(smpl_v, smpl_neutral_v, reduction='none').reshape(b, -1).mean(-1) + F.l1_loss(smpl_joints, smpl_neutral_joints, reduction='none').reshape(b, -1).mean(-1)
            loss_male = loss * (1 - is_female.reshape(b, 1, 1))

            loss = loss_female.mean() + loss_male.mean()
            loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            l_str += ', {}: {:0.4f}'.format('loss', loss.item())
            loop.set_description(l_str)
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.1

    return smplx_betas_neutral.detach()


class BEHAVEAnnotationDataset():

    def __init__(self, dataset_metadata, for_aug):
        self.for_aug = for_aug
        self.dataset_metadata = dataset_metadata
        print('collect all frames...')
        if for_aug:
            self.all_img_ids = list(dataset_metadata.go_through_all_frames_aug())
        else:
            self.all_img_ids = list(dataset_metadata.go_through_all_frames(split='all'))
        print('total {} frames'.format(len(self.all_img_ids)))

    def __len__(self, ):
        return len(self.all_img_ids)


    def __getitem__(self, idx):
        img_id = self.all_img_ids[idx]

        if self.for_aug:
            day_id, sub_id, obj_name, inter_type, frame_id, avatar_id, cam_id = img_id.split('_')
        else:
            day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')

        if self.for_aug:
            gender = 'male'
        else:
            gender = self.dataset_metadata.get_sub_gender(img_id)

        is_female = gender == 'female'

        person_mask = cv2.imread(self.dataset_metadata.get_person_mask_path(img_id, self.for_aug), cv2.IMREAD_GRAYSCALE) / 255
        object_full_mask = cv2.imread(self.dataset_metadata.get_object_full_mask_path(img_id, self.for_aug), cv2.IMREAD_GRAYSCALE) / 255
        person_bb_xyxy = extract_bbox_from_mask(person_mask).astype(np.float32)
        object_bb_xyxy = extract_bbox_from_mask(object_full_mask).astype(np.float32)
        if not self.for_aug:
            object_bb_xyxy *= 2
        hoi_bb_xyxy = np.concatenate([
            np.minimum(person_bb_xyxy[:2], object_bb_xyxy[:2]),
            np.maximum(person_bb_xyxy[2:], object_bb_xyxy[2:])
        ], axis=0).astype(np.float32)

        if not self.for_aug:
            object_rotmat, object_trans = self.dataset_metadata.load_object_RT(img_id)

            smplh_params = self.dataset_metadata.load_smpl_params(img_id)

            cx, cy, fx, fy = self.dataset_metadata.cam_intrinsics[int(cam_id)]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        else:
            object_rotmat, object_trans = self.dataset_metadata.load_object_RT_aug(img_id)

            smplh_params = self.dataset_metadata.load_smpl_params_aug(img_id)

            cx, cy, fx, fy = self.dataset_metadata.cam_intrinsics_aug
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        smplh_trans = smplh_params['trans']
        smplh_betas = smplh_params['betas']
        smplh_pose = smplh_params['pose']

        obj_keypoints_3d = self.dataset_metadata.load_object_keypoints(obj_name)
        obj_keypoints_3d = np.matmul(obj_keypoints_3d, object_rotmat.T) + object_trans.reshape(1, 3)

        max_object_kps_num = self.dataset_metadata.object_max_keypoint_num
        object_kpts_3d_padded = np.ones((max_object_kps_num, 3), dtype=np.float32)
        obj_kps_num = obj_keypoints_3d.shape[0]
        object_kpts_3d_padded[:obj_kps_num, :] = obj_keypoints_3d

        return img_id, gender, person_bb_xyxy, object_bb_xyxy, hoi_bb_xyxy, smplh_trans, smplh_betas, smplh_pose, is_female, object_kpts_3d_padded, object_rotmat, object_trans, K


def preprocess_behave(args):
    if args.for_aug:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    annotation_list = []

    behave_metadata = BEHAVEMetaData(args.root_dir)
    smpl_male = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)

    dataset = BEHAVEAnnotationDataset(behave_metadata, args.for_aug)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, drop_last=False)

    for item in tqdm(dataloader, desc='Preprocess Annotations (BEHAVE)'):
        img_id, gender, person_bb_xyxy, object_bb_xyxy, hoi_bb_xyxy, smplh_trans, smplh_betas, smplh_pose, is_female, obj_keypoints_3d, object_rotmat, object_trans, K = item

        focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1).to(device)
        optical_center = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1).to(device)

        b = smplh_betas.shape[0]
        smplh_trans = smplh_trans.to(device)
        smplh_betas = smplh_betas.to(device)
        smplh_pose = smplh_pose.to(device)
        smplh_pose_rotmat = axis_angle_to_matrix(smplh_pose.reshape(b, -1, 3))

        if not args.for_aug:
            is_female = torch.tensor(is_female, dtype=torch.float32).to(device)
            smplh_betas_male = fit_smplh_male(smplh_betas, smplh_pose_rotmat, is_female) # this may repeat 4 times for 4 cameras ...
        else:
            smplh_betas_male = smplh_betas

        smpl_out = smpl_male(betas=smplh_betas_male,
                             body_pose=smplh_pose_rotmat[:, 1:22],
                             global_orient=smplh_pose_rotmat[:, :1],
                             transl=smplh_trans)
        joint_3d = smpl_out.joints.detach()
        joint_2d = perspective_projection(joint_3d, focal_length=focal_length, optical_center=optical_center)

        obj_keypoints_3d = obj_keypoints_3d.to(device)
        obj_keypoints_2d = perspective_projection(obj_keypoints_3d, focal_length=focal_length, optical_center=optical_center)

        hoi_rotmat = smplh_pose_rotmat[:, 0]
        hoi_trans = joint_3d[:, 0]

        object_rotmat = object_rotmat.to(device)
        object_trans = object_trans.to(device)
        object_rel_rotmat = torch.matmul(hoi_rotmat.transpose(2, 1), object_rotmat.float())
        object_rel_trans = torch.matmul(hoi_rotmat.transpose(2, 1), (object_trans.float() - hoi_trans).reshape(b, 3, 1)).reshape(b, 3)

        joint_3d = joint_3d.cpu().numpy()
        joint_2d = joint_2d.cpu().numpy()
        obj_keypoints_2d = obj_keypoints_2d.cpu().numpy()
        obj_keypoints_3d = obj_keypoints_3d.cpu().numpy()
        person_bb_xyxy = person_bb_xyxy.numpy()
        object_bb_xyxy = object_bb_xyxy.numpy()
        hoi_bb_xyxy = hoi_bb_xyxy.numpy()
        smplh_betas = smplh_betas.cpu().numpy()
        smplh_betas_male = smplh_betas_male.cpu().numpy()
        smplh_trans = smplh_trans.cpu().numpy()
        smplh_pose = smplh_pose.cpu().numpy()
        smplh_pose_rotmat = smplh_pose_rotmat.cpu().numpy()
        object_rotmat = object_rotmat.cpu().numpy()
        object_trans = object_trans.cpu().numpy()
        object_rel_rotmat = object_rel_rotmat.cpu().numpy()
        object_rel_trans = object_rel_trans.cpu().numpy()
        hoi_trans = hoi_trans.cpu().numpy()
        hoi_rotmat = hoi_rotmat.cpu().numpy()
        K = K.numpy()
        for i in range(b):
            annotation_list.append({
                'img_id': img_id[i],
                'aug': args.for_aug,
                'gender': gender[i],
                'person_bb_xyxy': person_bb_xyxy[i], # (4, )
                'object_bb_xyxy': object_bb_xyxy[i], # (4, )
                'hoi_bb_xyxy': hoi_bb_xyxy[i], # (4, )
                'smplh_betas': smplh_betas[i], # (10, )
                'smplh_betas_male': smplh_betas_male[i], # (10, )
                'smplh_theta': smplh_pose[i], # (156, )
                'smplh_pose_rotmat': smplh_pose_rotmat[i], # (52, 3, 3)
                'smplh_trans': smplh_trans[i], # (3, )
                'smplh_joints_3d': joint_3d[i], # (73, 3)
                'smplh_joints_2d': joint_2d[i], # (73, 2)
                'obj_keypoints_3d': obj_keypoints_3d[i], # (-1, 3)
                'obj_keypoints_2d': obj_keypoints_2d[i], # (-1, 2)
                'object_trans': object_trans[i], # (3, )
                'object_rotmat': object_rotmat[i], # (3, 3)
                'object_rel_trans': object_rel_trans[i], # (3, )
                'object_rel_rotmat': object_rel_rotmat[i], # (3, 3)
                'hoi_trans': hoi_trans[i], # (3, )
                'hoi_rotmat': hoi_rotmat[i], # (3, 3)
                'cam_K': K[i], # (3, 3)
            })
    
    out_dir = './data/datasets'
    os.makedirs(out_dir, exist_ok=True)
    if not args.for_aug:
        annotation_list_train, annotation_list_test = split_behave_annotations(annotation_list, behave_metadata)
        save_pickle(annotation_list_train, os.path.join(out_dir, 'behave_train_list.pkl'))
        save_pickle(annotation_list_test, os.path.join(out_dir, 'behave_test_list.pkl'))
    else:
        save_pickle(annotation_list, os.path.join(out_dir, 'behave_aug_data_list.pkl'))


def split_behave_annotations(annotation_list, behave_metadata):
    train_list, test_list = [], []
    for item in annotation_list:
        if behave_metadata.in_train_set(item['img_id']):
            train_list.append(item)
        else:
            test_list.append(item)
    return train_list, test_list


class InterCapAnnotationDataset():

    def __init__(self, dataset_metadata,):
        self.dataset_metadata = dataset_metadata
        self.cam_calibration = dataset_metadata.load_cam_calibration()
        print('collect all frames...')
        self.all_img_ids = list(dataset_metadata.go_through_all_frames(split='all', ))
        print('total {} frames'.format(len(self.all_img_ids)))


    def __len__(self, ):
        return len(self.all_img_ids)


    def __getitem__(self, idx):
        img_id = self.all_img_ids[idx]

        sub_id, obj_id, seq_id, cam_id, frame_id = self.dataset_metadata.parse_img_id(img_id)
        obj_name = self.dataset_metadata.OBJECT_IDX2NAME[obj_id]

        gender = self.dataset_metadata.SUBID_GENDER[sub_id]

        is_female = gender == 'female'

        try:
            person_mask = cv2.imread(self.dataset_metadata.get_person_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255
            object_full_mask = cv2.imread(self.dataset_metadata.get_object_full_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255
            person_bb_xyxy = extract_bbox_from_mask(person_mask).astype(np.float32)
            object_bb_xyxy = extract_bbox_from_mask(object_full_mask).astype(np.float32)
            object_bb_xyxy *= 2
            hoi_bb_xyxy = np.concatenate([
                np.minimum(person_bb_xyxy[:2], object_bb_xyxy[:2]),
                np.maximum(person_bb_xyxy[2:], object_bb_xyxy[2:])
            ], axis=0).astype(np.float32)

            object_rotmat, object_trans = self.dataset_metadata.load_object_RT(img_id)

            calitration = self.cam_calibration[cam_id]
            cam_R = np.array(calitration['R'])
            cam_R = R.from_rotvec(cam_R).as_matrix()
            cam_T = np.array(calitration['T'])
            cx, cy = calitration['c']
            fx, fy = calitration['f']
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            object_rotmat = np.matmul(cam_R, object_rotmat)
            object_trans = np.matmul(cam_R, object_trans) + cam_T

            smpl_params = self.dataset_metadata.load_smpl_params(img_id)
        except:
            print('Exception in loading data...')
            return self.__getitem__(np.random.randint(len(self)))


        smplx_betas = smpl_params['betas'].reshape(10, )
        smplx_global_pose = smpl_params['global_orient'].reshape(3, )
        smplx_global_rotmat = R.from_rotvec(smplx_global_pose).as_matrix()
        smplx_trans = smpl_params['transl'].reshape(3, )
        smplx_body_pose = smpl_params['body_pose'].reshape((21, 3))

        smplx_global_rotmat = np.matmul(cam_R, smplx_global_rotmat)
        smplx_global_pose = R.from_matrix(smplx_global_rotmat).as_rotvec()
        smplx_pose = np.concatenate([smplx_global_pose, smplx_body_pose.reshape(-1)], axis=0) # (66, )
        smplx_trans = np.matmul(cam_R, smplx_trans) + cam_T

        obj_keypoints_3d = self.dataset_metadata.load_object_keypoints(obj_name)
        obj_keypoints_3d = np.matmul(obj_keypoints_3d, object_rotmat.T) + object_trans.reshape(1, 3)

        max_object_kps_num = self.dataset_metadata.object_max_keypoint_num
        object_kpts_3d_padded = np.ones((max_object_kps_num, 3), dtype=np.float32)
        obj_kps_num = obj_keypoints_3d.shape[0]
        object_kpts_3d_padded[:obj_kps_num, :] = obj_keypoints_3d

        return img_id, gender, person_bb_xyxy, object_bb_xyxy, hoi_bb_xyxy, smplx_betas, smplx_trans, smplx_pose, cam_R, cam_T, object_kpts_3d_padded, is_female, object_rotmat, object_trans, K


def split_intercap_dataset(args):
    np.random.seed(7)
    random.seed(7)

    all_sequences = []
    sub_ids = sorted(os.listdir(os.path.join(args.root_dir, 'RGBD_Images')))
    for sub_id in sub_ids:
        for obj_id in os.listdir(os.path.join(args.root_dir, 'RGBD_Images', sub_id)):
            for seq_name in os.listdir(os.path.join(args.root_dir, 'RGBD_Images', sub_id, obj_id)):
                if 'Seg' not in seq_name:
                    continue
                seq_id = seq_name[-1]
                seq_full_id = '_'.join([sub_id, obj_id, seq_id])
                all_sequences.append(seq_full_id)
    random.shuffle(all_sequences)
    num_sequences = len(all_sequences)
    trainset_size = int(0.8 * num_sequences)
    datalist_train = all_sequences[:trainset_size]
    datalist_test = all_sequences[trainset_size:]
    print('Total size: {}, number of training samples: {}, number of test samples: {}'.format(num_sequences, len(datalist_train), len(datalist_test)))
    save_json({'train': datalist_train, 'test': datalist_test}, 'data/datasets/intercap_split.json')


def preprocess_intercap(args):

    split_intercap_dataset(args) # 178 for train, 45 for test

    device = torch.device('cuda')
    annotation_list = []

    intercap_metadata = InterCapMetaData(args.root_dir)
    smpl = SMPLXLayer(model_path='data/models/smplx', gender='neutral').to(device)

    dataset = InterCapAnnotationDataset(intercap_metadata,)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, drop_last=False)

    for item in tqdm(dataloader, desc='Preprocess Annotations (InterCap)'):
        img_id, gender, person_bb_xyxy, object_bb_xyxy, hoi_bb_xyxy, smplx_betas, smplx_trans, smplx_pose, cam_R, cam_T, obj_keypoints_3d, is_female, object_rotmat, object_trans, K = item

        focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1).to(device)
        optical_center = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1).to(device)

        b = smplx_betas.shape[0]
        smplx_betas = smplx_betas.float().to(device)
        smplx_trans = smplx_trans.float().to(device)
        smplx_pose = smplx_pose.float().to(device)
        smplx_pose_rotmat = axis_angle_to_matrix(smplx_pose.reshape(b, -1, 3))
        cam_R = cam_R.float().to(device)
        cam_T = cam_T.float().to(device)

        is_female = is_female.to(torch.float32).to(device)
        smplx_betas_neutral = fit_smplx_neutral(smplx_betas, smplx_pose_rotmat, is_female)

        smpl_out_org = smpl(betas=smplx_betas_neutral.to(device),
                        body_pose=smplx_pose_rotmat[:, 1:22].to(device),
                        global_orient=torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(b, 1, 1),
                        transl=0 * smplx_trans.to(device))
        J_0 = smpl_out_org.joints[:,0].detach().reshape(b, 3, 1)
        smplx_trans = smplx_trans + (torch.matmul(cam_R, J_0) - J_0).squeeze(-1)

        smpl_out = smpl(betas=smplx_betas_neutral.to(device),
                        body_pose=smplx_pose_rotmat[:, 1:22].to(device),
                        global_orient=smplx_pose_rotmat[:, :1].to(device),
                        transl=smplx_trans.to(device))
        joint_3d = smpl_out.joints.detach()

        joint_2d = perspective_projection(joint_3d, focal_length=focal_length, optical_center=optical_center)

        obj_keypoints_3d = obj_keypoints_3d.to(device)
        obj_keypoints_2d = perspective_projection(obj_keypoints_3d, focal_length=focal_length, optical_center=optical_center)

        hoi_rotmat = smplx_pose_rotmat[:, 0]
        hoi_trans = joint_3d[:, 0]

        object_rotmat = object_rotmat.to(device)
        object_trans = object_trans.to(device)
        object_rel_rotmat = torch.matmul(hoi_rotmat.transpose(2, 1), object_rotmat.float())
        object_rel_trans = torch.matmul(hoi_rotmat.transpose(2, 1), (object_trans.float() - hoi_trans).reshape(b, 3, 1)).reshape(b, 3, )

        joint_3d = joint_3d.cpu().numpy()
        joint_2d = joint_2d.cpu().numpy()
        obj_keypoints_2d = obj_keypoints_2d.cpu().numpy()
        obj_keypoints_3d = obj_keypoints_3d.cpu().numpy()
        person_bb_xyxy = person_bb_xyxy.numpy()
        object_bb_xyxy = object_bb_xyxy.numpy()
        hoi_bb_xyxy = hoi_bb_xyxy.numpy()
        smplx_betas = smplx_betas.cpu().numpy()
        smplx_betas_neutral = smplx_betas_neutral.cpu().numpy()
        smplx_trans = smplx_trans.cpu().numpy()
        smplx_pose = smplx_pose.cpu().numpy()
        smplx_pose_rotmat = smplx_pose_rotmat.cpu().numpy()
        object_rotmat = object_rotmat.cpu().numpy()
        object_trans = object_trans.cpu().numpy()
        object_rel_rotmat = object_rel_rotmat.cpu().numpy()
        object_rel_trans = object_rel_trans.cpu().numpy()
        hoi_trans = hoi_trans.cpu().numpy()
        hoi_rotmat = hoi_rotmat.cpu().numpy()
        K = K.numpy()
        for i in range(b):
            annotation_list.append({
                'img_id': img_id[i],
                'gender': gender[i],
                'person_bb_xyxy': person_bb_xyxy[i], # (4, )
                'object_bb_xyxy': object_bb_xyxy[i], # (4, )
                'hoi_bb_xyxy': hoi_bb_xyxy[i], # (4, )
                'smplx_betas': smplx_betas[i], # (10, )
                'smplx_betas_neutral': smplx_betas_neutral[i], # (10, )
                'smplh_theta': smplx_pose[i], # (66, )
                'smplx_pose_rotmat': smplx_pose_rotmat[i], # (22, 3, 3)
                'smplx_trans': smplx_trans[i], # (3, )
                'smplx_joints_3d': joint_3d[i], # (73, 3)
                'smplx_joints_2d': joint_2d[i], # (73, 2)
                'obj_keypoints_3d': obj_keypoints_3d[i], # (-1, 3)
                'obj_keypoints_2d': obj_keypoints_2d[i], # (-1, 2)
                'object_trans': object_trans[i], # (3, )
                'object_rotmat': object_rotmat[i], # (3, 3)
                'object_rel_trans': object_rel_trans[i], # (3, )
                'object_rel_rotmat': object_rel_rotmat[i], # (3, 3)
                'hoi_trans': hoi_trans[i], # (3, )
                'hoi_rotmat': hoi_rotmat[i], # (3, 3)
                'cam_K': K[i], # (3, 3)
            })
    out_dir = './data/datasets'
    os.makedirs(out_dir, exist_ok=True)
    annotation_list_train, annotation_list_test = split_intercap_annotations(annotation_list, intercap_metadata)
    save_pickle(annotation_list_train, os.path.join(out_dir, 'intercap_train_list.pkl'))
    save_pickle(annotation_list_test, os.path.join(out_dir, 'intercap_test_list.pkl'))


def split_intercap_annotations(annotation_list, intercap_metadata):
    train_list, test_list = [], []
    for item in annotation_list:
        if intercap_metadata.in_train_set(item['img_id']):
            train_list.append(item)
        else:
            test_list.append(item)
    return train_list, test_list 


class BEHAVEExtendAnnotationDataset():

    def __init__(self, dataset_metadata, sequence_name, all_frames, annotations):
        self.dataset_metadata = dataset_metadata
        print('collect all frames...')
        self.all_img_ids = all_frames
        print('total {} frames'.format(len(self.all_img_ids)))
        self.annotations = annotations
        self.sequence_name = sequence_name

    def __len__(self, ):
        return len(self.all_img_ids)


    def __getitem__(self, idx):
        img_id = self.all_img_ids[idx]

        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')

        gender = self.dataset_metadata.get_sub_gender(img_id)

        is_female = gender == 'female'

        person_mask = cv2.imread(self.dataset_metadata.get_person_mask_path(img_id, ), cv2.IMREAD_GRAYSCALE) / 255
        object_full_mask = cv2.imread(self.dataset_metadata.get_object_full_mask_path(img_id,), cv2.IMREAD_GRAYSCALE) / 255
        person_bb_xyxy = extract_bbox_from_mask(person_mask).astype(np.float32)
        object_bb_xyxy = extract_bbox_from_mask(object_full_mask).astype(np.float32)
        object_bb_xyxy *= 2
        hoi_bb_xyxy = np.concatenate([
            np.minimum(person_bb_xyxy[:2], object_bb_xyxy[:2]),
            np.maximum(person_bb_xyxy[2:], object_bb_xyxy[2:])
        ], axis=0).astype(np.float32)

        annotation = self.annotations['t0' + frame_id]
        obj_axis_angle = annotation['ob_pose']
        object_rotmat = R.from_rotvec(obj_axis_angle).as_matrix()
        object_trans = annotation['ob_trans']

        cam_R, cam_T = self.dataset_metadata.cam_RT_matrix[day_id][int(cam_id)]
        object_trans = np.matmul(cam_R.transpose(), object_trans - cam_T)
        object_rotmat = np.matmul(cam_R.transpose(), object_rotmat)

        annotation = self.annotations['t0' + frame_id]
        smplh_params = {k: v for k, v in annotation.items() if 'ob_' not in k}

        cx, cy, fx, fy = self.dataset_metadata.cam_intrinsics[int(cam_id)]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        smplh_trans = smplh_params['trans']
        smplh_betas = smplh_params['betas']
        smplh_pose = smplh_params['poses'].copy()
        global_pose = smplh_pose[:3]
        global_pose_rotmat = R.from_rotvec(global_pose).as_matrix()
        global_pose_rotmat = np.matmul(cam_R.transpose(), global_pose_rotmat)
        global_pose = R.from_matrix(global_pose_rotmat).as_rotvec()
        smplh_pose[:3] = global_pose

        smplh_trans = np.matmul(cam_R.transpose(), smplh_trans - cam_T)

        obj_keypoints_3d = self.dataset_metadata.load_object_keypoints(obj_name)
        obj_keypoints_3d = np.matmul(obj_keypoints_3d, object_rotmat.T) + object_trans.reshape(1, 3)

        max_object_kps_num = self.dataset_metadata.object_max_keypoint_num
        object_kpts_3d_padded = np.ones((max_object_kps_num, 3), dtype=np.float32)
        obj_kps_num = obj_keypoints_3d.shape[0]
        object_kpts_3d_padded[:obj_kps_num, :] = obj_keypoints_3d

        return img_id, gender, person_bb_xyxy, object_bb_xyxy, hoi_bb_xyxy, smplh_trans, smplh_betas, smplh_pose, is_female, object_kpts_3d_padded, object_rotmat, object_trans, K


def preprocess_behave_extend(args):
    device = torch.device('cuda')
    annotation_list = []

    behave_metadata = BEHAVEExtendMetaData(args.root_dir, preload_annotations=False)
    smpl_male = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)

    all_sequences = list(behave_metadata.go_through_all_sequences())
    for sequence_name in tqdm(all_sequences):

        all_frames = list(behave_metadata.go_through_sequence(sequence_name))

        try:
            annotations = behave_metadata.load_annotations(sequence_name)
        except:
            # some sequences may lack of annotations
            continue

        dataset = BEHAVEExtendAnnotationDataset(behave_metadata, sequence_name, all_frames, annotations)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, drop_last=False)

        for item in tqdm(dataloader, desc='Preprocess Annotations (BEHAVE-Extended)'):
            img_id, gender, person_bb_xyxy, object_bb_xyxy, hoi_bb_xyxy, smplh_trans, smplh_betas, smplh_pose, is_female, obj_keypoints_3d, object_rotmat, object_trans, K = item

            focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1).to(device)
            optical_center = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1).to(device)

            b = smplh_betas.shape[0]
            smplh_trans = smplh_trans.to(device)
            smplh_betas = smplh_betas.to(device)
            smplh_pose = smplh_pose.to(device)
            smplh_pose_rotmat = axis_angle_to_matrix(smplh_pose.reshape(b, -1, 3))

            is_female = torch.tensor(is_female, dtype=torch.float32).to(device)
            smplh_betas_male = fit_smplh_male(smplh_betas, smplh_pose_rotmat, is_female) # this may repeat 4 times for 4 cameras ...

            smpl_out = smpl_male(betas=smplh_betas_male,
                                 body_pose=smplh_pose_rotmat[:, 1:22],
                                 global_orient=smplh_pose_rotmat[:, :1],
                                 transl=smplh_trans)
            joint_3d = smpl_out.joints.detach()
            joint_2d = perspective_projection(joint_3d, focal_length=focal_length, optical_center=optical_center)

            obj_keypoints_3d = obj_keypoints_3d.to(device)
            obj_keypoints_2d = perspective_projection(obj_keypoints_3d, focal_length=focal_length, optical_center=optical_center)

            hoi_rotmat = smplh_pose_rotmat[:, 0]
            hoi_trans = joint_3d[:, 0]

            object_rotmat = object_rotmat.to(device)
            object_trans = object_trans.to(device)
            object_rel_rotmat = torch.matmul(hoi_rotmat.transpose(2, 1), object_rotmat.float())
            object_rel_trans = torch.matmul(hoi_rotmat.transpose(2, 1), (object_trans.float() - hoi_trans).reshape(b, 3, 1)).reshape(b, 3)

            joint_3d = joint_3d.cpu().numpy()
            joint_2d = joint_2d.cpu().numpy()
            obj_keypoints_2d = obj_keypoints_2d.cpu().numpy()
            obj_keypoints_3d = obj_keypoints_3d.cpu().numpy()
            person_bb_xyxy = person_bb_xyxy.numpy()
            object_bb_xyxy = object_bb_xyxy.numpy()
            hoi_bb_xyxy = hoi_bb_xyxy.numpy()
            smplh_betas = smplh_betas.cpu().numpy()
            smplh_betas_male = smplh_betas_male.cpu().numpy()
            smplh_trans = smplh_trans.cpu().numpy()
            smplh_pose = smplh_pose.cpu().numpy()
            smplh_pose_rotmat = smplh_pose_rotmat.cpu().numpy()
            object_rotmat = object_rotmat.cpu().numpy()
            object_trans = object_trans.cpu().numpy()
            object_rel_rotmat = object_rel_rotmat.cpu().numpy()
            object_rel_trans = object_rel_trans.cpu().numpy()
            hoi_trans = hoi_trans.cpu().numpy()
            hoi_rotmat = hoi_rotmat.cpu().numpy()
            K = K.numpy()
            for i in range(b):
                annotation_list.append({
                    'img_id': img_id[i],
                    'gender': gender[i],
                    'person_bb_xyxy': person_bb_xyxy[i], # (4, )
                    'object_bb_xyxy': object_bb_xyxy[i], # (4, )
                    'hoi_bb_xyxy': hoi_bb_xyxy[i], # (4, )
                    'smplh_betas': smplh_betas[i], # (10, )
                    'smplh_betas_male': smplh_betas_male[i], # (10, )
                    'smplh_theta': smplh_pose[i], # (156, )
                    'smplh_pose_rotmat': smplh_pose_rotmat[i], # (52, 3, 3)
                    'smplh_trans': smplh_trans[i], # (3, )
                    'smplh_joints_3d': joint_3d[i], # (73, 3)
                    'smplh_joints_2d': joint_2d[i], # (73, 2)
                    'obj_keypoints_3d': obj_keypoints_3d[i], # (-1, 3)
                    'obj_keypoints_2d': obj_keypoints_2d[i], # (-1, 2)
                    'object_trans': object_trans[i], # (3, )
                    'object_rotmat': object_rotmat[i], # (3, 3)
                    'object_rel_trans': object_rel_trans[i], # (3, )
                    'object_rel_rotmat': object_rel_rotmat[i], # (3, 3)
                    'hoi_trans': hoi_trans[i], # (3, )
                    'hoi_rotmat': hoi_rotmat[i], # (3, 3)
                    'cam_K': K[i], # (3, 3)
                })
    
    out_dir = './data/datasets'
    os.makedirs(out_dir, exist_ok=True)
    annotation_list_train, annotation_list_test = split_behave_annotations(annotation_list, behave_metadata)
    save_pickle(annotation_list_train, os.path.join(out_dir, 'behave_extend_train_list.pkl'))
    save_pickle(annotation_list_test, os.path.join(out_dir, 'behave_extend_test_list.pkl'))


def filter_annotations(file_name):
    annotation_train = load_pickle(file_name)
    annotation_filtered = []
    for anno in tqdm(annotation_train):
        annotation_filtered.append({
            'img_id': anno['img_id'],
            'hoi_bb_xyxy': anno['hoi_bb_xyxy'], # (4, )
            'smplh_betas_male': anno['smplh_betas_male'], # (10, )
            'smplh_pose_rotmat': anno['smplh_pose_rotmat'][:22], # (52, 3, 3)
            'smplh_joints_3d': anno['smplh_joints_3d'][:22], # (73, 3)
            'smplh_joints_2d': anno['smplh_joints_2d'][:22], # (73, 2)
            'obj_keypoints_3d': anno['obj_keypoints_3d'], # (-1, 3)
            'obj_keypoints_2d': anno['obj_keypoints_2d'], # (-1, 2)
            'object_rel_trans': anno['object_rel_trans'], # (3, )
            'object_rel_rotmat': anno['object_rel_rotmat'], # (3, 3)
            'object_trans': anno['object_trans'], # (3, )
            'object_rotmat': anno['object_rotmat'], # (3, 3)
            'hoi_trans': anno['hoi_trans'], # (3, )
            'hoi_rotmat': anno['hoi_rotmat'], # (3, 3)
            'cam_K': anno['cam_K'], # (3, 3)

        })
    save_pickle(annotation_filtered, file_name.replace('.pkl', '_filtered.pkl'))


def main(args):
    if args.is_behave:
        preprocess_behave(args)
    elif args.behave_extend:
        preprocess_behave_extend(args)
    else:
        preprocess_intercap(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str, help='Dataset root directory.')
    parser.add_argument('--is_behave', default=False, action='store_true', help='Process behave dataset or intercap dataset.')
    parser.add_argument('--behave_extend', default=False, action='store_true', help='Process behave dataset or intercap dataset.')
    parser.add_argument('--for_aug', default=False, action='store_true', help='For BEHAVE viewport-free augmented dataset. (Only Valid for BEHAVE dataset)')
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()

    main(args)
