import os
import cv2
import random
import numpy as np
import torch
from yacs.config import CfgNode as CN
from smplx import SMPLH
from pytorch3d.transforms import axis_angle_to_matrix

from prohmr.datasets.utils import do_augmentation, generate_image_patch, rot_aa, keypoint_3d_processing, trans_point2d

from stackflow.datasets.data_utils import load_pickle, load_mask
from stackflow.datasets.behave_metadata import load_object_mesh_templates, get_img_path_from_id, IMG_HEIGHT, IMG_WIDTH


class BEHAVEImageDataset(object):

    def __init__(self, cfg, exlusion_occlusion=False, is_train=True):
        self.is_train = is_train
        self.smpl = SMPLH(model_path=cfg.data.smplh_path, gender='male', ext='pkl')

        if is_train:
            data_list = load_pickle(cfg.dataset.behave.datalist_train_real)
            if cfg.dataset.behave.use_fake:
                data_list.extend(load_pickle(cfg.dataset.behave.datalist_fake))
        else:
            data_list = load_pickle(cfg.dataset.behave.datalist_test_real)

        self.data_list = data_list
        # if exlusion_occlusion:
        #     self.data_list = [item for item in self.data_list if item['object_visible_ratio'] > 0.3]
        # self.data_list = [item for item in self.data_list if item['object_visible_ratio'] < 0.3]

        self.object_templates = load_object_mesh_templates()
        self.bg_aug_ratio = cfg.dataset.bg_aug_ratio
        bg_dir = cfg.dataset.bg_dir
        self.background_list = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)]
        self.rescale_ratio = cfg.dataset.rescale_ratio

        self.img_size = cfg.dataset.img_size
        self.mean = 255. * np.array(cfg.dataset.mean)
        self.std = 255. * np.array(cfg.dataset.std)
        self.augm_config = self.get_aug_config(cfg)


    def get_aug_config(self, cfg):
        _C = CN(new_allowed=True)
        _C.SCALE_FACTOR = cfg.dataset.scale_factor
        _C.ROT_FACTOR = cfg.dataset.rot_factor
        _C.TRANS_FACTOR = cfg.dataset.trans_factor
        _C.COLOR_SCALE = cfg.dataset.color_scale
        _C.ROT_AUG_RATE = cfg.dataset.rot_aug_rate
        _C.DO_FLIP = False
        _C.FLIP_AUG_RATE = 0
        _C.EXTREME_CROP_AUG_RATE = 0
        return _C


    @staticmethod
    def load_bg_im(im_real, bg_list):
        h, w, c = im_real.shape
        bg_num = len(bg_list)
        idx = random.randint(0, bg_num - 1)
        bg_path = os.path.join(bg_list[idx])
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
        return bg_im


    def load_image(self, image_file):
        _avatar_id = ''
        avatar_ids = ['00032_shortlong', '00032_shortshort', '00096_shortlong', '00096_shortshort', '00159_shortlong', '00159_shortshort', '03223_shortlong', '03223_shortshort']
        for avatar_id in avatar_ids:
            if avatar_id in image_file:
                _avatar_id = avatar_id
        avatar_id = avatar_ids[np.random.randint(8)]
        if os.path.exists(image_file.replace(_avatar_id, avatar_id)):
            image_file = image_file.replace(_avatar_id, avatar_id)
        cvimg = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cvimg = cv2.resize(cvimg, (0, 0),  fx=self.rescale_ratio, fy=self.rescale_ratio, interpolation=cv2.INTER_LINEAR) # rescale original image to speed up data processing
        return cvimg


    def change_bg(self, rgb, msk):
        if self.background_list is None:
            return rgb
        bg_im = self.load_bg_im(rgb, self.background_list)
        msk = msk.astype(np.bool)
        bg_im[msk] = rgb[msk]
        return bg_im


    def get_example(self, cvimg, mask, center_x, center_y, width, height, global_orient, joint_2d, joint_3d, patch_width, patch_height, mean, std, do_augment, augm_config):
        img_height, img_width, img_channels = cvimg.shape
        img_size = np.array([img_height, img_width])

        if do_augment:
            scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty = do_augmentation(augm_config)
        else:
            scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty = 1.0, 0, False, False, [1.0, 1.0, 1.0], 0., 0.
        do_flip = False # flip cause some non-reasonable results

        center_x += width * tx
        center_y += height * ty

        img_patch_cv, trans = generate_image_patch(cvimg, center_x, center_y, width, height, patch_width, patch_height, do_flip, scale, rot)
        if mask is not None:
            mask, _ = generate_image_patch(mask, center_x, center_y, width, height, patch_width, patch_height, do_flip, scale, rot)
            img_patch_cv = self.change_bg(img_patch_cv, mask)

        img_patch = img_patch_cv[:, :, ::-1].astype(np.float32)
        img_patch = img_patch.transpose((2, 0, 1))

        global_orient = rot_aa(global_orient, rot)
        joint_3d = keypoint_3d_processing(joint_3d, None, rot, do_flip)
        for n_jt in range(len(joint_2d)):
            joint_2d[n_jt, 0:2] = trans_point2d(joint_2d[n_jt, 0:2], trans)
        joint_2d[:, :-1] = joint_2d[:, :-1] / patch_width - 0.5

        for n_c in range(img_channels):
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
            if mean is not None and std is not None:
                img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
        return img_patch, global_orient, joint_2d, joint_3d, img_size, (center_x, center_y), scale
        

    def __len__(self, ):
        return len(self.data_list)


    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_id = item['img_id']
        data_type = item['data_type']
        image_file = get_img_path_from_id(img_id=img_id, data_type=data_type)
        image = self.load_image(image_file)

        mask = None
        if np.random.random() < self.bg_aug_ratio:
            if data_type == 'real':
                mask_path = image_file.replace('color', 'person_mask')
            else:
                mask_path = image_file.replace('hoi_color', 'person_mask')
            person_mask = load_mask(mask_path) / 255
            person_mask = cv2.resize(person_mask, (0, 0), fx=self.rescale_ratio, fy=self.rescale_ratio, interpolation=cv2.INTER_NEAREST)
            if data_type == 'real':
                mask_path = image_file.replace('color', 'obj_rend_mask')
            else:
                mask_path = image_file.replace('hoi_color', 'object_mask')
            try:
                object_mask = load_mask(mask_path) / 255
            except:
                object_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH))
            object_mask = cv2.resize(object_mask, (0, 0), fx=self.rescale_ratio, fy=self.rescale_ratio, interpolation=cv2.INTER_NEAREST)
            mask = person_mask.astype(np.bool) | object_mask.astype(np.bool)
            mask = np.dstack([mask, mask, mask]).astype(np.float32)

        person_bbox = item['person_bb_xyxy'] * self.rescale_ratio
        object_bbox = item['object_bb_xyxy'] * self.rescale_ratio
        x1, y1 = min(person_bbox[0], object_bbox[0]), min(person_bbox[1], object_bbox[1])
        x2, y2 = max(person_bbox[2], object_bbox[2]), max(person_bbox[3], object_bbox[3])
        joint_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        bbox_size = max(y2 - y1, x2 - x1)
        bbox_size = bbox_size * 1.2
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        joint_3d = np.array(item['smplh_joints_3d']).astype(np.float32)
        joint_2d = np.array(item['smplh_joints_2d']).astype(np.float32) * self.rescale_ratio
        joint_3d = np.concatenate([joint_3d, np.ones((73, 1))], axis=1)
        joint_2d = np.concatenate([joint_2d, np.ones((73, 1))], axis=1)

        img_patch, global_orient, joint_2d, joint_3d, img_size, box_centers, scale = self.get_example(image, mask, center_x, center_y, bbox_size, bbox_size, item['smplh_pose'][:3], joint_2d, joint_3d,
            self.img_size, self.img_size, self.mean, self.std, self.is_train, self.augm_config)

        bbox_size = bbox_size * scale / self.rescale_ratio # the bbox size in original image (unscaled image)
        box_centers = np.array(box_centers, dtype=np.float32) / self.rescale_ratio
        K = item['cam_intrinsics']

        # get smpl vertices and object vertices
        smplh_betas = torch.tensor(item['smplh_betas'], dtype=torch.float32).reshape(1, -1)
        smplh_pose = torch.tensor(item['smplh_pose'], dtype=torch.float32).reshape(1, -1)
        smplh_trans = torch.tensor(item['smplh_trans'], dtype=torch.float32).reshape(1, -1)
        smplh_out = self.smpl(betas=smplh_betas, body_pose=smplh_pose[:, 3:66], global_orient=0*smplh_pose[:, :3], transl=0*smplh_trans)
        smplh_vertices = smplh_out.vertices.detach().reshape(-1, 3)

        root_rotmat = axis_angle_to_matrix(smplh_pose[:, :3]).reshape(1, 3, 3)
        J = smplh_out.joints.detach()
        root_T = smplh_trans - (torch.einsum('bst,bvt->bvs', root_rotmat, J[:, 0:1]) - J[:, 0:1]).squeeze(1)
        root_T = root_T.reshape(3, )

        object_rotmat = torch.tensor(item['object_rotmat'], dtype=torch.float32).reshape(3, 3)
        object_trans = torch.tensor(item['object_trans'], dtype=torch.float32).reshape(3, )
        object_rotmat = root_rotmat[0].transpose(1, 0) @ object_rotmat
        object_trans = root_rotmat[0].transpose(1, 0) @ (object_trans - root_T).reshape(3, 1)

        obj_name = img_id.split('_')[2]
        object_verts_org, _ = self.object_templates[obj_name]
        max_points = 1700
        object_verts_org = torch.cat([object_verts_org,
                                      torch.zeros(max_points - object_verts_org.shape[0], 3, dtype=object_verts_org.dtype)], dim=0)
        object_vertices = torch.einsum('st,vt->vs', object_rotmat, object_verts_org) + object_trans.reshape(1, 3)

        results = {}
        results['img_id'] = img_id
        results['image'] = img_patch.astype(np.float32) # (3, 256, 256)
        results['box_size'] = bbox_size
        results['scale'] = scale
        results['box_center'] = box_centers # (2, )
        results['optical_center'] = np.array([K[0, 2], K[1, 2]], dtype=np.float32)
        results['focal_length'] = np.array([K[0, 0], K[1, 1]], dtype=np.float32)
        results['K'] = np.array(K, dtype=np.float32)
        results['joint_2d'] = joint_2d
        results['joint_3d'] = joint_3d
        results['smpl_betas'] = item['smplh_betas'].reshape(-1, )
        results['smpl_pose'] = item['smplh_pose'].reshape(-1, )[:66]
        results['smpl_global_orient'] = global_orient.astype(np.float32)
        results['smpl_trans'] = smplh_trans.reshape(3, )
        results['smpl_v_orig'] = smplh_vertices
        results['object_v_orig'] = object_vertices
        results['object_label'] = item['object_label']
        results['object_rotmat'] = object_rotmat.reshape(3, 3)
        results['object_trans'] = object_trans.reshape(-1)
        return results


    def get_crop_object_image(self, idx):
        item = self.data_list[idx]
        img_id = item['img_id']
        image_file = get_img_path_from_id(img_id=img_id)
        image = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        box = item['object_bb_xyxy']
        x1, y1, x2, y2 = box
        c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        s = max(x2 - x1, y2 - y1) * 1.5

        import sys
        sys.path.append('externals/EPro-PnP/EPro-PnP-6DoF/lib')
        from utils.img import zoom_in

        try:
            img_patch, c_h, c_w, s = zoom_in(image, c, s, 256)
        except:
            img_patch = zoom_in(image, c, s, 256)
            c_h = c_w = 0
            s = 0
        img_patch = img_patch.transpose(2, 0, 1).astype(np.float32) / 255.
        c = np.array([c_w, c_h])

        return img_patch, s, c, item['object_label'], item['cam_intrinsics'], item['img_id']


    def get_person_crop(self, idx):
        item = self.data_list[idx]
        img_id = item['img_id']
        data_type = item['data_type']
        image_file = get_img_path_from_id(img_id=img_id, data_type=data_type)
        image = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        h, w = image.shape[:2]
        person_bbox = item['person_bb_xyxy']
        cx, cy = (person_bbox[0] + person_bbox[2]) / 2, (person_bbox[1] + person_bbox[3]) / 2
        s = max(person_bbox[2] - person_bbox[0], person_bbox[3] - person_bbox[1])
        s = s * 1.2

        s = int(s)

        image_out = np.zeros((s, s, 3))

        x1, y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        x2, y2 = int(min(x1 + s, w)), int(min(y1 + s, h))
        image_out[: y2 - y1, : x2 - x1, :] = image[y1:y2, x1:x2, :]
        try:
            scaled_img = cv2.resize(image_out, (256, 256), interpolation=cv2.INTER_LINEAR)
        except:
            scaled_img = np.zeros((256, 256, 3))

        center = np.array([x1 + s / 2, y1 + s / 2])
        bbox_size = s

        return scaled_img, bbox_size, center
