import os
import cv2
import numpy as np

from stackflow.datasets.utils import (
    load_pickle, 
    load_json,
    get_augmentation_params, 
    generate_image_patch, 
    rot_keypoints, 
    get_rotmat_from_angle, 
    trans_keypoints,
    rotate_2d
)
from stackflow.datasets.behave_metadata import BEHAVEMetaData


class BEHAVEDataset(object):

    def __init__(self, cfg, is_train=True, for_evaluation=False):
        self.cfg = cfg
        self.is_train = is_train
        self.for_evaluation = for_evaluation

        if is_train:
            self.annotation_list = load_pickle(cfg.dataset.annotation_file_train)
            if self.cfg.dataset.with_aug_data:
                self.aug_annotation_list = load_pickle(cfg.dataset.annotation_file_aug)
        else:
            self.annotation_list = load_pickle(cfg.dataset.annotation_file_test)

        self.dataset_metadata = BEHAVEMetaData(cfg.dataset.root_dir)
        self.mean = cfg.dataset.mean
        self.std = cfg.dataset.std

        bg_dir = cfg.dataset.bg_dir
        self.background_list = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)]


    def __len__(self, ):
        return len(self.annotation_list)


    def __getitem__(self, idx):
        if not self.for_evaluation:
            return self.get_item_for_training(idx)
        else:
            return self.get_item_for_evaluation(idx)


    def change_bg(self, image, img_id, for_aug):
        # from CDPN (https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi)
        h, w, c = image.shape

        bg_num = len(self.background_list)
        idx = np.random.randint(0, bg_num - 1)
        bg_path = os.path.join(self.background_list[idx])
        bg_im = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        bg_h, bg_w, bg_c = bg_im.shape
        real_hw_ratio = float(h) / float(w)
        bg_hw_ratio = float(bg_h) / float(bg_w)
        if real_hw_ratio <= bg_hw_ratio:
            crop_w = bg_w
            crop_h = int(bg_w * real_hw_ratio)
        else:
            crop_h = bg_h 
            crop_w = int(bg_h / bg_hw_ratio)
        bg_im = bg_im[:crop_h, :crop_w, :]
        bg_im = cv2.resize(bg_im, (w, h), interpolation=cv2.INTER_LINEAR)

        person_mask = cv2.imread(self.dataset_metadata.get_person_mask_path(img_id, for_aug=for_aug), cv2.IMREAD_GRAYSCALE) / 255
        object_full_mask_path = self.dataset_metadata.get_object_full_mask_path(img_id, for_aug=for_aug)
        object_full_mask = cv2.imread(object_full_mask_path, cv2.IMREAD_GRAYSCALE) / 255
        if not for_aug:
            object_full_mask = cv2.resize(object_full_mask, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        mask = (person_mask.astype(np.bool_) | object_full_mask.astype(np.bool_))

        bg_im[mask] = image[mask]
        return bg_im


    def get_item_for_training(self, idx):
        if self.is_train and self.cfg.dataset.with_aug_data and np.random.rand() < self.cfg.dataset.aug_ratio:
            annotation = self.aug_annotation_list[np.random.randint(len(self.aug_annotation_list))]
        else:
            annotation = self.annotation_list[idx]
        img_id = annotation['img_id']
        obj_name = img_id.split('_')[2]

        image_path = self.dataset_metadata.get_image_path(img_id, for_aug=annotation['aug'])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self.is_train and np.random.random() < self.cfg.dataset.change_bg_ratio:
            image = self.change_bg(image, img_id, for_aug=annotation['aug'])

        hoi_bb_xyxy = annotation['hoi_bb_xyxy']
        box_width, box_height = hoi_bb_xyxy[2:] - hoi_bb_xyxy[:2]
        box_size = max(box_width, box_height)
        box_size = box_size * (self.cfg.dataset.hoi_img_padding_ratio + 1)
        box_center_x, box_center_y = (hoi_bb_xyxy[2:] + hoi_bb_xyxy[:2]) / 2

        if self.is_train:
            tx, ty, rot, scale, color_scale = get_augmentation_params(self.cfg)
        else:
            tx, ty, rot, scale, color_scale = 0., 0., 0., 1., [1., 1., 1.]

        box_center_x += tx * box_width
        box_center_y += ty * box_width
        out_size = self.cfg.dataset.img_size
        box_size = max(1, box_size * scale)

        img_patch, img_trans = generate_image_patch(image, box_center_x, box_center_y, box_size, out_size, rot, color_scale)
        img_patch = img_patch[:, :, ::-1].astype(np.float32)
        img_patch = img_patch.transpose((2, 0, 1))

        for n_c in range(3):
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        person_joint_3d = annotation['smplh_joints_3d'][:22]
        object_kpts_3d = annotation['obj_keypoints_3d']
        person_joint_3d = rot_keypoints(person_joint_3d, -rot)
        object_kpts_3d = rot_keypoints(object_kpts_3d, -rot)

        person_joint_2d = annotation['smplh_joints_2d'][:22]
        object_kpts_2d = annotation['obj_keypoints_2d']
        person_joint_2d = trans_keypoints(person_joint_2d, img_trans)
        object_kpts_2d = trans_keypoints(object_kpts_2d,  img_trans)
        person_joint_2d = person_joint_2d / out_size - 0.5
        object_kpts_2d = object_kpts_2d / out_size - 0.5

        max_object_kps_num = self.dataset_metadata.object_max_keypoint_num
        object_kpts_3d_padded = np.zeros((max_object_kps_num, 3), dtype=np.float32)
        object_kpts_2d_padded = np.zeros((max_object_kps_num, 2), dtype=np.float32)
        object_kpts_weights = np.zeros((max_object_kps_num, ), dtype=np.float32)
        obj_kps_num = self.dataset_metadata.object_num_keypoints[obj_name]
        object_kpts_2d_padded[:obj_kps_num, :] = object_kpts_2d[:obj_kps_num, :]
        object_kpts_3d_padded[:obj_kps_num, :] = object_kpts_3d[:obj_kps_num, :]
        object_kpts_weights[:obj_kps_num, ] = 1

        smplh_pose_rotmat = annotation['smplh_pose_rotmat']
        global_pose_rotmat = smplh_pose_rotmat[0]
        aug_rotmat = get_rotmat_from_angle(-rot)
        global_pose_rotmat = np.matmul(aug_rotmat, global_pose_rotmat)
        body_pose_rotmat = smplh_pose_rotmat[1:22]
        smpl_pose_rotmat = np.concatenate([global_pose_rotmat.reshape(1, 3, 3), body_pose_rotmat], axis=0)

        cam_K = annotation['cam_K']
        optical_center = np.array([cam_K[0, 2], cam_K[1, 2]], dtype=np.float32)
        focal_length = np.array([cam_K[0, 0], cam_K[1, 1]], dtype=np.float32)

        rot_center_x = annotation['smplh_joints_2d'][0, 0]
        rot_center_y = annotation['smplh_joints_2d'][0, 1]
        rot_rad = - np.pi * rot / 180
        smpl_trans_2d = np.array([box_center_x, box_center_y]) + rotate_2d(np.array([rot_center_x - box_center_x, rot_center_y - box_center_y]), rot_rad)
        smpl_trans_3d = annotation['smplh_joints_3d'][0]
        smpl_trans_3d[0] = (smpl_trans_2d[0] - optical_center[0]) / focal_length[0] * smpl_trans_3d[2]
        smpl_trans_3d[1] = (smpl_trans_2d[1] - optical_center[1]) / focal_length[1] * smpl_trans_3d[2]

        results = {}
        results['img_id'] = img_id
        results['image'] = img_patch
        results['box_size'] = box_size
        results['box_center'] = np.array([box_center_x, box_center_y], dtype=np.float32) # [2, ]
        results['optical_center'] = optical_center # [2, ]
        results['focal_length'] = focal_length # [2, ]

        results['person_joint_3d'] = person_joint_3d # [22, 3]
        results['object_kpts_3d'] = object_kpts_3d_padded # [16, 3]
        results['person_joint_2d'] = person_joint_2d # [22, 2]
        results['object_kpts_2d'] = object_kpts_2d_padded # [16, 2]
        results['object_kpts_weights'] = object_kpts_weights # [16, ]
        results['smpl_pose_rotmat'] = smpl_pose_rotmat.astype(np.float32) # [22, 3, 3]
        results['smpl_betas'] = annotation['smplh_betas_male'].astype(np.float32) # [10, ]
        results['smpl_trans'] = smpl_trans_3d.astype(np.float32) # [3, ]

        results['object_rel_rotmat'] = annotation['object_rel_rotmat'].astype(np.float32) # [3, 3]
        results['object_rel_trans'] = annotation['object_rel_trans'].astype(np.float32) # [3, ]
        results['object_labels'] = self.dataset_metadata.OBJECT_NAME2IDX[obj_name]
        results['hoi_rotmat'] = annotation['hoi_rotmat'].astype(np.float32) # [3, 3]
        results['hoi_trans'] = annotation['hoi_trans'].astype(np.float32) # [3, ]

        return results


    def get_item_for_evaluation(self, idx):
        annotation = self.annotation_list[idx]
        img_id = annotation['img_id']
        obj_name = img_id.split('_')[2]

        image_path = self.dataset_metadata.get_image_path(img_id, for_aug=annotation['aug'])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        #####################################################################################################
        # Here, we assume that the ground-truth bounding boxes have been given to compare fairly.
        hoi_bb_xyxy = annotation['hoi_bb_xyxy']
        box_width, box_height = hoi_bb_xyxy[2:] - hoi_bb_xyxy[:2]
        box_size = max(box_width, box_height)
        box_size = box_size * (self.cfg.dataset.hoi_img_padding_ratio + 1)
        box_center_x, box_center_y = (hoi_bb_xyxy[2:] + hoi_bb_xyxy[:2]) / 2
        #####################################################################################################

        tx, ty, rot, scale, color_scale = 0., 0., 0., 1., [1., 1., 1.]

        box_center_x += tx * box_width
        box_center_y += ty * box_width
        out_size = self.cfg.dataset.img_size
        box_size = max(1, box_size * scale)

        img_patch, img_trans = generate_image_patch(image, box_center_x, box_center_y, box_size, out_size, rot, color_scale)
        img_patch = img_patch[:, :, ::-1].astype(np.float32)
        img_patch = img_patch.transpose((2, 0, 1))

        for n_c in range(3):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        cam_K = annotation['cam_K']
        optical_center = np.array([cam_K[0, 2], cam_K[1, 2]], dtype=np.float32)
        focal_length = np.array([cam_K[0, 0], cam_K[1, 1]], dtype=np.float32)

        results = {}
        results['img_id'] = img_id
        results['image'] = img_patch
        results['box_size'] = np.array(box_size, dtype=np.float32) # (1, )
        results['box_center'] = np.array([box_center_x, box_center_y], dtype=np.float32) # [2, ]
        results['optical_center'] = optical_center # [2, ]
        results['focal_length'] = focal_length # [2, ]
        #####################################################################################################
        # Here, we assume that the object labels have been given to compare fairly.
        results['object_labels'] = self.dataset_metadata.OBJECT_NAME2IDX[obj_name]
        #####################################################################################################

        epro_pnp_coor = self.dataset_metadata.get_pred_coor_map_path(img_id)
        epro_pnp_coor = load_pickle(epro_pnp_coor)
        obj_x3d = epro_pnp_coor['x3d']
        obj_x2d = epro_pnp_coor['x2d']
        obj_w2d = epro_pnp_coor['w2d']
        results['obj_x3d'] = obj_x3d.astype(np.float32)
        results['obj_x2d'] = obj_x2d.astype(np.float32)
        results['obj_w2d'] = obj_w2d.astype(np.float32)

        openpose_path = self.dataset_metadata.get_openpose_path(img_id)
        try:
            openpose = load_json(openpose_path)
            keypoints = openpose['people'][0]['pose_keypoints_2d']
            keypoints = np.array(keypoints).reshape((25, 3)).astype(np.float32)
        except: # file may not exists or no person is detected
            keypoints = np.zeros((25, 3), dtype=np.float32)
        results['person_kps'] = keypoints

        return results