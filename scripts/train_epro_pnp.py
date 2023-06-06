import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', 'externals', 'EPro-PnP', 'EPro-PnP-6DoF'))
sys.path.insert(0, os.path.join(file_dir, '..', 'externals', 'EPro-PnP', 'EPro-PnP-6DoF', 'lib'))
sys.path.insert(0, os.path.join(file_dir, '..', ))
import cv2
import time
import argparse
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from progress.bar import Bar
from tensorboardX import SummaryWriter
from scipy.spatial.transform import Rotation

from stackflow.datasets.behave_metadata import BEHAVEMetaData
from stackflow.datasets.intercap_metadata import InterCapMetaData
from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from stackflow.datasets.utils import load_pickle

from lib.config import get_base_config
from lib.model import build_model, save_model
from lib.datasets.lm import LM
from lib.utils.img import zoom_in, im_norm_255
from lib.utils.transform3d import prj_vtx_cam
from lib.utils.utils import AverageMeter
from lib.ops.rotation_conversions import matrix_to_quaternion
from lib.ops.pnp.camera import PerspectiveCamera
from lib.ops.pnp.cost_fun import AdaptiveHuberPnPCost
from lib.ops.pnp.levenberg_marquardt import LMSolver, RSLMSolver
from lib.ops.pnp.epropnp import EProPnP6DoF


class BEHAVEObjectDataset(LM):

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.root_dir = cfg.dataset.root_dir
        self.split = split

        self.behave_metadata = BEHAVEMetaData(self.root_dir)
        self.infos = self.load_model_info()
        self.frame_list = list(self.behave_metadata.go_through_all_frames(split=split))

        bg_dir = cfg.dataset.bg_dir
        self.background_list = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)]


    def load_model_info(self, ):
        infos = {}
        meta_data = self.behave_metadata
        for obj_name in meta_data.OBJECT_COORDINATE_NORM.keys():
            obj_id = meta_data.OBJECT_NAME2IDX[obj_name]
            xyz_min = meta_data.OBJECT_COORDINATE_NORM[obj_name]
            infos[obj_id] = {}
            infos[obj_id]['min_x'] = xyz_min[0]
            infos[obj_id]['min_y'] = xyz_min[1]
            infos[obj_id]['min_z'] = xyz_min[2]
        return infos


    def change_bg(self, image, mask):
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

        bg_im[mask] = image[mask]
        return bg_im


    def __len__(self, ):
        return len(self.frame_list)


    def __getitem__(self, idx):
        # this data loading process may be a bit heavy
        if self.split == 'train':
            img_id = self.frame_list[idx]
            day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.behave_metadata.parse_img_id(img_id)
            obj_id = self.behave_metadata.OBJECT_NAME2IDX[obj_name]

            object_rotmat, object_trans = self.behave_metadata.load_object_RT(img_id)
            pose = np.concatenate([object_rotmat, object_trans.reshape((3, 1))], axis=1)

            rgb = cv2.imread(self.behave_metadata.get_image_path(img_id))
            person_mask = cv2.imread(self.behave_metadata.get_person_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255
            object_render_mask_path = self.behave_metadata.get_object_render_mask_path(img_id)
            object_render_mask = cv2.imread(object_render_mask_path, cv2.IMREAD_GRAYSCALE) / 255

            # downsampling to speed up
            rgb = cv2.resize(rgb, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            person_mask = cv2.resize(person_mask, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            object_render_mask = cv2.resize(object_render_mask, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            mask = object_render_mask - (person_mask.astype(np.bool_) & object_render_mask.astype(np.bool_))
            _h, _w, _ = rgb.shape
            if object_render_mask is None:
                object_render_mask = np.zeros((_h, _w))

            if np.random.random() < 0.5:
                rgb = self.change_bg(rgb, mask.astype(np.bool_))

            object_full_mask = cv2.imread(object_render_mask_path.replace('sequences', 'object_coor_maps').replace('obj_rend_mask.jpg', 'mask_full.jpg'), cv2.IMREAD_GRAYSCALE) / 255
            if object_full_mask is None:
                object_full_mask = np.zeros((_h, _w))

            if object_full_mask.sum() == 0:
                visible_ratio = 0
            else:
                visible_ratio = max(0, mask.sum() / (object_full_mask.sum()))

            cx, cy, fx, fy = self.behave_metadata.cam_intrinsics[int(cam_id)]
            cam_K = np.array([[fx / 2, 0, cx / 2], [0, fy / 2, cy / 2], [0, 0, 1]])

            try:
                ys, xs = np.nonzero(object_full_mask)
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                box = np.array([x1, y1, x2 - x1, y2 - y1])
            except: # invisiable object
                box = np.array([0, 0, 256, 256])

            coor = np.zeros((_h, _w, 3)).astype(np.float32)
            coor_load = load_pickle(object_render_mask_path.replace('sequences', 'object_coor_maps').replace('obj_rend_mask.jpg', 'obj_coor.pkl'))
            try:
                u = coor_load['u']
                l = coor_load['l']
                h = coor_load['h']
                w = coor_load['w']
                coor[u:(u+h),l:(l+w),:] = coor_load['coor']
            except:
                pass

            if self.cfg.dataiter.dzi:
                c, s = self.xywh_to_cs_dzi(box, self.cfg.augment.pad_ratio, s_max=max(_h, _w))
            else:
                c, s = self.xywh_to_cs(box, self.cfg.augment.pad_ratio, s_max=max(_h, _w))

            if self.cfg.dataiter.denoise_coor:
                coor = self.denoise_coor(coor)

            rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
            rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
            msk, *_ = zoom_in(mask, c, s, self.cfg.dataiter.out_res, channel=1)
            coor, *_ = zoom_in(coor, c, s, self.cfg.dataiter.out_res, interpolate=cv2.INTER_NEAREST)
            c = np.array([c_w_, c_h_])
            s = s_

            coor = self.norm_coor(coor, obj_id).transpose(2, 0, 1)
            inp = rgb
            out = np.concatenate([coor, msk[None, :, :]], axis=0)
            loss_msk = np.stack([msk, msk, msk, np.ones_like(msk)], axis=0)
            trans = pose[:, 3]
            c_obj, _ = prj_vtx_cam(trans, cam_K)
            c_delta = self.c_rel_delta(c_obj, c, box[2:])
            d_local = self.d_scaled(trans[2], s, self.cfg.dataiter.out_res)
            trans_local = np.append(c_delta, [d_local], axis=0)

            return obj_name, obj_id, inp, out, loss_msk, trans_local, pose, c, s, np.asarray(box), cam_K.astype(np.float32), visible_ratio


class InterCapObjectDataset(LM):

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.root_dir = cfg.dataset.root_dir
        self.split = split

        self.intercap_metadata = InterCapMetaData(self.root_dir)
        self.infos = self.load_model_info()
        self.frame_list = list(self.intercap_metadata.go_through_all_frames(split=split))
        self.annotations = load_pickle('./data/datasets/intercap_train_list.pkl')
        self.annotations = {item['img_id']: item for item in self.annotations}

        bg_dir = cfg.dataset.bg_dir
        self.background_list = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)]


    def load_model_info(self, ):
        infos = {}
        meta_data = self.intercap_metadata
        for obj_name in meta_data.OBJECT_COORDINATE_NORM.keys():
            obj_id = int(meta_data.OBJECT_NAME2IDX[obj_name]) - 1
            xyz_min = meta_data.OBJECT_COORDINATE_NORM[obj_name]
            infos[obj_id] = {}
            infos[obj_id]['min_x'] = xyz_min[0]
            infos[obj_id]['min_y'] = xyz_min[1]
            infos[obj_id]['min_z'] = xyz_min[2]
        return infos


    def change_bg(self, image, mask):
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

        bg_im[mask] = image[mask]
        return bg_im


    def __len__(self, ):
        return len(self.frame_list)


    def __getitem__(self, idx):
        try:
            return self.load_data(idx)
        except:
            print('Exception found during loading data.')
            idx = np.random.randint(len(self))
            return self.__getitem__(idx)


    def load_data(self, idx):
        # this data loading process may be a bit heavy
        if self.split == 'train':
            img_id = self.frame_list[idx]
            sub_id, obj_id, seq_id, cam_id, frame_id = self.intercap_metadata.parse_img_id(img_id)
            obj_name = self.intercap_metadata.OBJECT_IDX2NAME[obj_id]
            obj_id = int(self.intercap_metadata.OBJECT_NAME2IDX[obj_name]) - 1

            object_rotmat = self.annotations[img_id]['object_rotmat']
            object_trans = self.annotations[img_id]['object_trans']

            calitration = self.intercap_metadata.cam_calibration[cam_id]
            cx, cy = calitration['c']
            fx, fy = calitration['f']
            pose = np.concatenate([object_rotmat, object_trans.reshape((3, 1))], axis=1)

            rgb = cv2.imread(self.intercap_metadata.get_image_path(img_id))
            person_mask = cv2.imread(self.intercap_metadata.get_person_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255

            object_full_mask_path = self.intercap_metadata.get_object_full_mask_path(img_id)
            object_full_mask = cv2.imread(object_full_mask_path, cv2.IMREAD_GRAYSCALE)

            # downsampling to speed up
            rgb = cv2.resize(rgb, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            person_mask = cv2.resize(person_mask, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            _h, _w, _ = rgb.shape

            object_full_mask = object_full_mask / 255
            mask = object_full_mask - (person_mask.astype(np.bool_) & object_full_mask.astype(np.bool_))

            if np.random.random() < 0.5:
                rgb = self.change_bg(rgb, mask.astype(np.bool_))

            if object_full_mask.sum() == 0:
                visible_ratio = 0
            else:
                visible_ratio = max(0, mask.sum() / (object_full_mask.sum()))

            cam_K = np.array([[fx / 2, 0, cx / 2], [0, fy / 2, cy / 2], [0, 0, 1]])

            ys, xs = np.nonzero(object_full_mask)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            box = np.array([x1, y1, x2 - x1, y2 - y1])

            coor = np.zeros((_h, _w, 3)).astype(np.float32)
            coor_load = load_pickle(self.intercap_metadata.get_object_coor_path(img_id))
            u = coor_load['u']
            l = coor_load['l']
            h = coor_load['h']
            w = coor_load['w']
            coor[u:(u+h),l:(l+w),:] = coor_load['coor']

            if self.cfg.dataiter.dzi:
                c, s = self.xywh_to_cs_dzi(box, self.cfg.augment.pad_ratio, s_max=max(_h, _w))
            else:
                c, s = self.xywh_to_cs(box, self.cfg.augment.pad_ratio, s_max=max(_h, _w))

            if self.cfg.dataiter.denoise_coor:
                coor = self.denoise_coor(coor)

            rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
            rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
            msk, *_ = zoom_in(mask, c, s, self.cfg.dataiter.out_res, channel=1)
            coor, *_ = zoom_in(coor, c, s, self.cfg.dataiter.out_res, interpolate=cv2.INTER_NEAREST)
            c = np.array([c_w_, c_h_])
            s = s_

            coor = self.norm_coor(coor, obj_id).transpose(2, 0, 1)
            inp = rgb
            out = np.concatenate([coor, msk[None, :, :]], axis=0)
            loss_msk = np.stack([msk, msk, msk, np.ones_like(msk)], axis=0)
            trans = pose[:, 3]
            c_obj, _ = prj_vtx_cam(trans, cam_K)
            c_delta = self.c_rel_delta(c_obj, c, box[2:])
            d_local = self.d_scaled(trans[2], s, self.cfg.dataiter.out_res)
            trans_local = np.append(c_delta, [d_local], axis=0)

            return obj_name, obj_id, inp, out, loss_msk, trans_local, pose, c, s, np.asarray(box), cam_K.astype(np.float32), visible_ratio


class BEHAVEExtendObjectDataset(LM):

    def __init__(self, cfg, annotations, split):
        self.cfg = cfg
        self.root_dir = cfg.dataset.root_dir
        self.split = split

        self.behave_metadata = BEHAVEExtendMetaData(self.root_dir, preload_annotations=False)
        self.infos = self.load_model_info()
        self.frame_list = list(self.behave_metadata.go_through_all_frames(split=split))
        self.annotations = annotations

        bg_dir = cfg.dataset.bg_dir
        self.background_list = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)]


    def load_model_info(self, ):
        infos = {}
        meta_data = self.behave_metadata
        for obj_name in meta_data.OBJECT_COORDINATE_NORM.keys():
            obj_id = meta_data.OBJECT_NAME2IDX[obj_name]
            xyz_min = meta_data.OBJECT_COORDINATE_NORM[obj_name]
            infos[obj_id] = {}
            infos[obj_id]['min_x'] = xyz_min[0]
            infos[obj_id]['min_y'] = xyz_min[1]
            infos[obj_id]['min_z'] = xyz_min[2]
        return infos


    def change_bg(self, image, mask):
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

        bg_im[mask] = image[mask]
        return bg_im


    def __len__(self, ):
        return len(self.frame_list)


    def __getitem__(self, idx):
        try:
            return self.load_data(idx)
        except:
            print('Exception raised during data loading.')
            return self.load_data(np.random.randint(len(self)))


    def load_data(self, idx):
        # this data loading process may be a bit heavy
        if self.split == 'train':
            img_id = self.frame_list[idx]
            day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.behave_metadata.parse_img_id(img_id)
            obj_id = self.behave_metadata.OBJECT_NAME2IDX[obj_name]

            object_rotmat = self.annotations[img_id]['object_rotmat']
            object_trans = self.annotations[img_id]['object_trans']
            pose = np.concatenate([object_rotmat, object_trans.reshape((3, 1))], axis=1)

            rgb = cv2.imread(self.behave_metadata.get_image_path(img_id))
            person_mask = cv2.imread(self.behave_metadata.get_person_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255

            object_full_mask_path = self.behave_metadata.get_object_full_mask_path(img_id)
            object_full_mask = cv2.imread(object_full_mask_path, cv2.IMREAD_GRAYSCALE)

            # downsampling to speed up
            rgb = cv2.resize(rgb, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            person_mask = cv2.resize(person_mask, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            _h, _w, _ = rgb.shape

            object_full_mask = object_full_mask / 255
            try:
                mask = object_full_mask - (person_mask.astype(np.bool_) & object_full_mask.astype(np.bool_))
            except:
                mask = np.zeros((_h, _w))

            if np.random.random() < 0.5:
                rgb = self.change_bg(rgb, mask.astype(np.bool_))
            
            if object_full_mask.sum() == 0:
                visible_ratio = 0
            else:
                visible_ratio = max(0, mask.sum() / (object_full_mask.sum()))

            cx, cy, fx, fy = self.behave_metadata.cam_intrinsics[int(cam_id)]
            cam_K = np.array([[fx / 2, 0, cx / 2], [0, fy / 2, cy / 2], [0, 0, 1]])

            try:
                ys, xs = np.nonzero(object_full_mask)
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                box = np.array([x1, y1, x2 - x1, y2 - y1])
            except: # invisiable object
                box = np.array([0, 0, 256, 256])

            coor = np.zeros((_h, _w, 3)).astype(np.float32)
            coor_load = load_pickle(self.behave_metadata.get_object_coor_path(img_id))
            try:
                u = coor_load['u']
                l = coor_load['l']
                h = coor_load['h']
                w = coor_load['w']
                coor[u:(u+h),l:(l+w),:] = coor_load['coor']
            except:
                pass

            if self.cfg.dataiter.dzi:
                c, s = self.xywh_to_cs_dzi(box, self.cfg.augment.pad_ratio, s_max=max(_h, _w))
            else:
                c, s = self.xywh_to_cs(box, self.cfg.augment.pad_ratio, s_max=max(_h, _w))

            if self.cfg.dataiter.denoise_coor:
                coor = self.denoise_coor(coor)

            rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
            rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
            msk, *_ = zoom_in(mask, c, s, self.cfg.dataiter.out_res, channel=1)
            coor, *_ = zoom_in(coor, c, s, self.cfg.dataiter.out_res, interpolate=cv2.INTER_NEAREST)
            c = np.array([c_w_, c_h_])
            s = s_

            coor = self.norm_coor(coor, obj_id).transpose(2, 0, 1)
            inp = rgb
            out = np.concatenate([coor, msk[None, :, :]], axis=0)
            loss_msk = np.stack([msk, msk, msk, np.ones_like(msk)], axis=0)
            trans = pose[:, 3]
            c_obj, _ = prj_vtx_cam(trans, cam_K)
            c_delta = self.c_rel_delta(c_obj, c, box[2:])
            d_local = self.d_scaled(trans[2], s, self.cfg.dataiter.out_res)
            trans_local = np.append(c_delta, [d_local], axis=0)

            return obj_name, obj_id, inp, out, loss_msk, trans_local, pose, c, s, np.asarray(box), cam_K.astype(np.float32), visible_ratio
        elif self.split == 'test':
            img_id = self.frame_list[idx]
            day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.behave_metadata.parse_img_id(img_id)
            obj_id = self.behave_metadata.OBJECT_NAME2IDX[obj_name]

            object_rotmat, object_trans = self.behave_metadata.load_object_RT(img_id)
            cam_R, cam_T = self.behave_metadata.cam_RT_matrix[day_id][int(cam_id)]
            object_trans = np.matmul(cam_R.transpose(), object_trans - cam_T)
            object_rotmat = np.matmul(cam_R.transpose(), object_rotmat)
            pose = np.concatenate([object_rotmat, object_trans.reshape((3, 1))], axis=1)

            rgb = cv2.imread(self.behave_metadata.get_image_path(img_id))
            person_mask = cv2.imread(self.behave_metadata.get_person_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255

            object_full_mask_path = self.behave_metadata.get_object_full_mask_path(img_id)
            object_full_mask = cv2.imread(object_full_mask_path, cv2.IMREAD_GRAYSCALE)

            # downsampling to speed up
            rgb = cv2.resize(rgb, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            person_mask = cv2.resize(person_mask, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            _h, _w, _ = rgb.shape

            object_full_mask = object_full_mask / 255
            mask = object_full_mask - (person_mask.astype(np.bool_) & object_full_mask.astype(np.bool_))

            if object_full_mask.sum() == 0:
                visible_ratio = 0
            else:
                visible_ratio = max(0, mask.sum() / (object_full_mask.sum()))

            cx, cy, fx, fy = self.behave_metadata.cam_intrinsics[int(cam_id)]
            cam_K = np.array([[fx / 2, 0, cx / 2], [0, fy / 2, cy / 2], [0, 0, 1]])

            h, w = object_full_mask.shape
            ys, xs = np.nonzero(object_full_mask)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            box = np.array([x1, y1, x2 - x1, y2 - y1]) / 2

            c, s = self.xywh_to_cs(box, self.cfg.augment.pad_ratio, s_max=max(self.cfg.dataset.img_w / 2, self.cfg.dataset.img_h / 2))

            rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
            rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
            c = np.array([c_w_, c_h_])
            s = s_

            inp = rgb
            trans = pose[:, 3]
            c_obj, _ = prj_vtx_cam(trans, cam_K)
            c_delta = self.c_rel_delta(c_obj, c, box[2:])
            d_local = self.d_scaled(trans[2], s, self.cfg.dataiter.out_res)
            trans_local = np.append(c_delta, [d_local], axis=0)

            return obj_name, obj_id, inp, pose, c, s, np.asarray(box), trans_local, cam_K.astype(np.float32), visible_ratio


class MonteCarloPoseLoss(nn.Module):

    def __init__(self, init_norm_factor=1.0, momentum=0.01):
        super(MonteCarloPoseLoss, self).__init__()
        self.register_buffer('norm_factor', torch.tensor(init_norm_factor, dtype=torch.float))
        self.momentum = momentum

    def forward(self, pose_sample_logweights, cost_target, norm_factor):
        """
        Args:
            pose_sample_logweights: Shape (mc_samples, num_obj)
            cost_target: Shape (num_obj, )
            norm_factor: Shape ()
        """
        if self.training:
            with torch.no_grad():
                self.norm_factor.mul_(
                    1 - self.momentum).add_(self.momentum * norm_factor)

        loss_tgt = cost_target
        loss_pred = torch.logsumexp(pose_sample_logweights, dim=0)  # (num_obj, )

        loss_pose = loss_tgt + loss_pred  # (num_obj, )
        loss_pose[torch.isnan(loss_pose)] = 0
        loss_pose = loss_pose / self.norm_factor

        return loss_pose


def train(epoch, cfg, data_loader, model, obj_info, criterions, optimizer=None):
    model.train()
    Loss = AverageMeter()
    Loss_rot = AverageMeter()
    Loss_trans = AverageMeter()
    Loss_mc = AverageMeter()
    Loss_t = AverageMeter()
    Loss_r = AverageMeter()
    Loss_w2d = AverageMeter()
    Norm_factor = AverageMeter()
    Grad_norm = AverageMeter()
    num_iters = len(data_loader)
    bar = Bar('{}'.format(cfg.pytorch.exp_id[-60:]), max=num_iters)

    vis_dir = os.path.join(cfg.pytorch.save_path, 'train_vis_{}'.format(epoch))
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    epropnp = EProPnP6DoF(
        mc_samples=512,
        num_iter=4,
        solver=LMSolver(
            dof=6,
            num_iter=5,
            init_solver=RSLMSolver(
                dof=6,
                num_points=16,
                num_proposals=4,
                num_iter=3))).cuda(cfg.pytorch.gpu)
    monte_carlo_pose_loss = MonteCarloPoseLoss().cuda(cfg.pytorch.gpu)

    for i, (obj_name, obj_id, inp, target, loss_msk, trans_local, pose, c_box, s_box, box, cam_K, visible_ratio) in enumerate(data_loader):
        cur_iter = i + (epoch - 1) * num_iters
        if cfg.pytorch.gpu > -1:
            inp_var = inp.cuda(cfg.pytorch.gpu).float()
            target_var = target.cuda(cfg.pytorch.gpu).float()
            loss_msk_var = loss_msk.cuda(cfg.pytorch.gpu).float()
            trans_local_var = trans_local.cuda(cfg.pytorch.gpu).float()
            pose_var = pose.cuda(cfg.pytorch.gpu).float()
            c_box_var = c_box.cuda(cfg.pytorch.gpu).float()
            s_box_var = s_box.cuda(cfg.pytorch.gpu).float()
            cam_K = cam_K.cuda(cfg.pytorch.gpu).float()
            visible_ratio = visible_ratio.cuda(cfg.pytorch.gpu).float()
        else:
            inp_var = inp.float()
            target_var = target.float()
            loss_msk_var = loss_msk.float()
            trans_local_var = trans_local.float()
            pose_var = pose.float()
            c_box_var = c_box.float()
            s_box_var = s_box.float()

        bs = len(inp)
        # forward propagation
        T_begin = time.time()
        # import ipdb; ipdb.set_trace()
        (noc, w2d, scale), pred_trans = model(inp_var)
        T_end = time.time() - T_begin

        if i % cfg.test.disp_interval == 0:
            # display input image
            inp_rgb = (inp[0].cpu().numpy().copy() * 255)[::-1, :, :].astype(np.uint8)
            cv2.imwrite(os.path.join(vis_dir, '{}_inp.png'.format(i)), inp_rgb.transpose(1,2,0)[:, :, ::-1])
            if 'rot' in cfg.pytorch.task.lower():
                # display coordinates map
                pred_coor = noc[0].data.cpu().numpy().copy()
                pred_coor[0] = im_norm_255(pred_coor[0])
                pred_coor[1] = im_norm_255(pred_coor[1])
                pred_coor[2] = im_norm_255(pred_coor[2])
                pred_coor = np.asarray(pred_coor, dtype=np.uint8)
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_x_pred.png'.format(i)), pred_coor[0])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_y_pred.png'.format(i)), pred_coor[1])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_z_pred.png'.format(i)), pred_coor[2])
                gt_coor = target[0, 0:3].data.cpu().numpy().copy()
                gt_coor[0] = im_norm_255(gt_coor[0])
                gt_coor[1] = im_norm_255(gt_coor[1])
                gt_coor[2] = im_norm_255(gt_coor[2])
                gt_coor = np.asarray(gt_coor, dtype=np.uint8)
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_x_gt.png'.format(i)), gt_coor[0])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_y_gt.png'.format(i)), gt_coor[1])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_z_gt.png'.format(i)), gt_coor[2])
                # display confidence map
                pred_conf = w2d[0].reshape(2, -1).softmax(dim=-1)
                pred_conf = pred_conf.mean(dim=0).reshape(64, 64).data.cpu().numpy().copy()
                pred_conf = (im_norm_255(pred_conf)).astype(np.uint8)
                cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred.png'.format(i)), pred_conf)
            if 'trans' in cfg.pytorch.task.lower():
                pred_trans_ = pred_trans[0].data.cpu().numpy().copy()
                gt_trans_ = trans_local[0].data.cpu().numpy().copy()
                cfg.writer.add_scalar('train_trans_x_gt', gt_trans_[0], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_y_gt', gt_trans_[1], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_z_gt', gt_trans_[2], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_x_pred', pred_trans_[0], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_y_pred', pred_trans_[1], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_z_pred', pred_trans_[2], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_x_err', pred_trans_[0]-gt_trans_[0], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_y_err', pred_trans_[1]-gt_trans_[1], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_z_err', pred_trans_[2]-gt_trans_[2], i + (epoch-1) * num_iters)

        loss_weights = visible_ratio ** (0.5)
        # loss
        if 'rot' in cfg.pytorch.task.lower() and not cfg.network.rot_head_freeze:
            dim = [[abs(obj_info[obj_id_]['min_x']),
                    abs(obj_info[obj_id_]['min_y']),
                    abs(obj_info[obj_id_]['min_z'])] for obj_id_ in obj_id.cpu().numpy()]
            dim = noc.new_tensor(dim) # (n, 3)
            x3d = noc * dim[..., None, None] # (bs, 3, h, w)

            s = s_box_var.to(torch.int64)
            wh_begin = c_box_var.to(torch.int64) - s[:, None] / 2. # (n, 2)
            wh_unit = s.to(torch.float32) / cfg.dataiter.out_res  # (n, )

            wh_arange = torch.arange(cfg.dataiter.out_res, device=noc.device, dtype=torch.float32)
            y, x = torch.meshgrid(wh_arange, wh_arange)  # (h, w)
            # (bs, 2, h, w)
            x2d = torch.stack((wh_begin[:, 0, None, None] + x * wh_unit[:, None, None],
                               wh_begin[:, 1, None, None] + y * wh_unit[:, None, None]), dim=1)
            rot_mat = pose_var[:, :, :3]  # (bs, 3, 3)
            trans_vec = pose_var[:, :, 3]  # (bs, 3)
            rot_quat = matrix_to_quaternion(rot_mat)
            pose_gt = torch.cat((trans_vec, rot_quat), dim=-1)

            sample_pts = [np.random.choice(64 * 64, size=64 * 64 // 8, replace=False) for _ in range(bs)]
            sample_inds = x2d.new_tensor(sample_pts, dtype=torch.int64)
            batch_inds = torch.arange(bs, device=x2d.device)[:, None]
            x3d = x3d.flatten(2).transpose(-1, -2)[batch_inds, sample_inds]
            x2d = x2d.flatten(2).transpose(-1, -2)[batch_inds, sample_inds]
            w2d = w2d.flatten(2).transpose(-1, -2)[batch_inds, sample_inds]
            mask2d = loss_msk[:, :2].flatten(2).transpose(-1, -2)[batch_inds, sample_inds]
            # Due to a legacy design decision, we use an alternative to standard softmax, i.e., normalizing
            # the mean before exponential map.
            # w2d = (w2d - w2d.mean(dim=1, keepdim=True) - math.log(w2d.size(1))).exp() * scale[:, None, :]
            w2d = w2d.sigmoid() * scale[:, None, :]

            allowed_border = 30 * wh_unit  # (n, )
            camera = PerspectiveCamera(
                cam_mats=cam_K,
                z_min=0.01,
                lb=wh_begin - allowed_border[:, None],
                ub=wh_begin + (cfg.dataiter.out_res - 1) * wh_unit[:, None] + allowed_border[:, None])
            cost_fun = AdaptiveHuberPnPCost(
                relative_delta=0.1)
            cost_fun.set_param(x2d, w2d)
            _, _, pose_opt_plus, _, pose_sample_logweights, cost_tgt = epropnp.monte_carlo_forward(
                x3d, x2d, w2d, camera, cost_fun,
                pose_init=pose_gt, force_init_solve=True, with_pose_opt_plus=True)

            loss_mc = monte_carlo_pose_loss(
                pose_sample_logweights, cost_tgt, scale.detach().mean())
            loss_mc = (loss_mc * loss_weights).mean()

            loss_t = (pose_opt_plus[:, :3] - pose_gt[:, :3]).norm(dim=-1)
            beta = 0.05
            loss_t = torch.where(loss_t < beta, 0.5 * loss_t.square() / beta,
                                 loss_t - 0.5 * beta)
            loss_t = (loss_weights * loss_t).mean()

            dot_quat = (pose_opt_plus[:, None, 3:] @ pose_gt[:, 3:, None]).squeeze(-1).squeeze(-1)
            loss_r = (1 - dot_quat.square()) * 2
            loss_r = (loss_weights * loss_r).mean()

            loss_rot = criterions[cfg.loss.rot_loss_type](
                loss_msk_var[:, :3] * noc, loss_msk_var[:, :3] * target_var[:, :3])
            loss_rot = (loss_weights[:, None, None, None] * loss_rot).mean()
            loss_w2d = F.l1_loss(w2d[mask2d == 0], torch.zeros_like(w2d[mask2d == 0]))
        else:
            loss_mc = 0
            loss_t = 0
            loss_r = 0
            loss_rot = 0

        if 'trans' in cfg.pytorch.task.lower() and not cfg.network.trans_head_freeze:
            loss_trans = criterions[cfg.loss.trans_loss_type](pred_trans, trans_local_var)
            loss_trans = (loss_weights[:, None], loss_trans).mean()
        else:
            loss_trans = 0

        loss = cfg.loss.rot_loss_weight * loss_rot + cfg.loss.trans_loss_weight * loss_trans \
               + cfg.loss.mc_loss_weight * loss_mc + cfg.loss.t_loss_weight * loss_t \
               + cfg.loss.r_loss_weight * loss_r + loss_w2d

        Loss.update(loss.item() if loss != 0 else 0, bs)
        Loss_rot.update(loss_rot.item() if loss_rot != 0 else 0, bs)
        Loss_trans.update(loss_trans.item() if loss_trans != 0 else 0, bs)
        Loss_mc.update(loss_mc.item() if loss_mc != 0 else 0, bs)
        Loss_t.update(loss_t.item() if loss_t != 0 else 0, bs)
        Loss_r.update(loss_r.item() if loss_r != 0 else 0, bs)
        Loss_w2d.update(loss_w2d.item() if loss_w2d != 0 else 0, bs)
        Norm_factor.update(model.monte_carlo_pose_loss.norm_factor.item(), bs)

        cfg.writer.add_scalar('data/loss', loss.item() if loss != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_rot', loss_rot.item() if loss_rot != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_trans', loss_trans.item() if loss_trans != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_mc', loss_mc.item() if loss_mc != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_t', loss_t.item() if loss_t != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_r', loss_r.item() if loss_r != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_w2d', loss_w2d.item() if loss_w2d != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/norm_factor', model.monte_carlo_pose_loss.norm_factor.item(), cur_iter)

        optimizer.zero_grad()
        model.zero_grad()
        T_begin = time.time()
        loss.backward()

        grad_norm = []
        for p in model.parameters():
            if (p.grad is None) or (not p.requires_grad):
                continue
            else:
                grad_norm.append(torch.norm(p.grad.detach()))
        grad_norm = torch.norm(torch.stack(grad_norm))
        Grad_norm.update(grad_norm.item(), bs)
        cfg.writer.add_scalar('data/grad_norm', grad_norm.item(), cur_iter)

        if not torch.isnan(grad_norm).any():
            optimizer.step()

        Bar.suffix = 'train Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | ' \
                     'Loss {loss.avg:.4f} | Loss_rot {loss_rot.avg:.4f} | ' \
                     'Loss_trans {loss_trans.avg:.4f} | Loss_mc {loss_mc.avg:.4f} | ' \
                     'Norm_factor {norm_factor.avg:.4f} | Grad_norm {grad_norm.avg:.4f} | ' \
                     'Loss_t {loss_t.avg:.4f} | Loss_r {loss_r.avg:.4f} | Loss_w2d {loss_w2d.avg:.4f}'.format(
            epoch, i, num_iters, total=bar.elapsed_td, eta=bar.eta_td,
            loss=Loss, loss_rot=Loss_rot, loss_trans=Loss_trans, loss_mc=Loss_mc,
            norm_factor=Norm_factor, grad_norm=Grad_norm, loss_t=Loss_t, loss_r=Loss_r, loss_w2d=Loss_w2d)
        if i % 10 == 0:
            print(Bar.suffix)
            sys.stdout.flush()
        bar.next()
    bar.finish()


def update_config(cfg, args):
    cfg.pytorch.exp_id = 'epropnp_behave' if args.is_behave else 'epropnp_intercap'
    cfg.pytorch.task = 'rot'
    cfg.pytorch.threads_num = 16
    cfg.pytorch.save_mode = 'last'
    file_dir = os.path.dirname(__file__)
    cfg.pytorch.load_model = os.path.join(file_dir, '../outputs/epropnp/behave_extend/model_last.checkpoint')
    if args.is_behave:
        cfg.pytorch.save_path = os.path.join(file_dir, '../outputs/epropnp/behave')
    elif args.behave_extend:
        cfg.pytorch.save_path = os.path.join(file_dir, '../outputs/epropnp/behave_extend')
    else:
        cfg.pytorch.save_path = os.path.join(file_dir, '../outputs/epropnp/intercap')
    cfg.pytorch.gpu = 0

    cfg.pytorch['tensorboard'] = os.path.join(cfg.pytorch.save_path, 'tensorboard')
    if not os.path.exists(cfg.pytorch.tensorboard):
        os.makedirs(cfg.pytorch.tensorboard)
    cfg.writer = SummaryWriter(cfg.pytorch.tensorboard)

    cfg.dataset.root_dir = args.root_dir
    cfg.dataset.bg_dir = '/public/home/huochf/datasets/LineMOD/bg_images/VOC2012/JPEGImages/'

    cfg.network.numBackLayers = 50
    cfg.network.rot_output_channels = 5
    cfg.train.train_batch_size = 32

    cfg.train.begin_epoch = 0
    if args.is_behave:
        cfg.train.end_epoch = 50
        cfg.train.lr_epoch_step = [40, 45, 50]
    elif args.behave_extend:
        cfg.train.end_epoch = 10
        cfg.train.lr_epoch_step = [8, 9, 10]
    else:
        cfg.train.end_epoch = 30
        cfg.train.lr_epoch_step = [20, 25, 30]
    cfg.test.disp_interval = 1000


def main(args):
    cfg = get_base_config()
    update_config(cfg, args)
    network, optimizer = build_model(cfg)
    criterions = {'L1': torch.nn.L1Loss(reduction='none'),
                  'L2': torch.nn.MSELoss(reduction='none')}

    if cfg.pytorch.gpu > -1:
        network = network.cuda(cfg.pytorch.gpu)
        for k in criterions.keys():
            criterions[k] = criterions[k].cuda(cfg.pytorch.gpu)

    def _worker_init_fn():
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)

    if args.is_behave:
        dataset = BEHAVEObjectDataset(cfg, 'train')
    elif args.behave_extend:
        print('loading annotations ...')
        annotations = load_pickle('./data/datasets/behave_extend_train_list_filtered.pkl')
        annotations_shared = {}
        for item in annotations:
            annotations_shared[item['img_id']] = item
        dataset = BEHAVEExtendObjectDataset(cfg, annotations_shared, 'train')
    else:
        dataset = InterCapObjectDataset(cfg, 'train')
    obj_info = dataset.load_model_info()

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=int(cfg.pytorch.threads_num),
        worker_init_fn=_worker_init_fn()
    )

    for epoch in range(cfg.train.begin_epoch, cfg.train.end_epoch + 1):
        mark = epoch if (cfg.pytorch.save_mode == 'all') else 'last'
        train(epoch, cfg, train_loader, network, obj_info, criterions, optimizer)
        save_model(os.path.join(cfg.pytorch.save_path, 'model_{}.checkpoint'.format(mark)), network)  # optimizer

        if epoch in cfg.train.lr_epoch_step:
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= cfg.train.lr_factor
                    print("drop lr to {}".format(param_group['lr']))

    torch.save(network.cpu(), os.path.join(cfg.pytorch.save_path, 'model_cpu.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str, help='Dataset root directory.')
    parser.add_argument('--is_behave', default=False, action='store_true', help='Process behave dataset or intercap dataset.')
    parser.add_argument('--behave_extend', default=False, action='store_true', help='Train on BEHAVE-Extended dataset.')
    args = parser.parse_args()

    main(args)
