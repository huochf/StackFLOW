import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', 'externals', 'EPro-PnP', 'EPro-PnP-6DoF'))
sys.path.insert(0, os.path.join(file_dir, '..', 'externals', 'EPro-PnP', 'EPro-PnP-6DoF', 'lib'))
sys.path.insert(0, os.path.join(file_dir, '..', ))
import numpy as np
from tqdm import tqdm
import cv2
import time
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from progress.bar import Bar
from tensorboardX import SummaryWriter
from scipy.spatial.transform import Rotation

from lib.config import get_base_config
from lib.model import build_model, save_model
from lib.datasets.lm import LM
from lib.utils.img import zoom_in, im_norm_255
from lib.ops.pnp.camera import PerspectiveCamera
from lib.ops.pnp.cost_fun import AdaptiveHuberPnPCost
from lib.ops.pnp.levenberg_marquardt import LMSolver, RSLMSolver
from lib.ops.pnp.epropnp import EProPnP6DoF

from stackflow.datasets.behave_metadata import BEHAVEMetaData
from stackflow.datasets.intercap_metadata import InterCapMetaData
from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from stackflow.datasets.utils import save_pickle


class BEHAVEObjectDataset(LM):

    def __init__(self, cfg, split='test'):
        assert split == 'test'
        self.cfg = cfg
        self.root_dir = cfg.dataset.root_dir
        self.split = split

        self.dataset_metadata = BEHAVEMetaData(self.root_dir)
        self.infos = self.load_model_info()
        self.frame_list = list(self.dataset_metadata.go_through_all_frames(split=split))


    def load_model_info(self, ):
        infos = {}
        meta_data = self.dataset_metadata
        for obj_name in meta_data.OBJECT_COORDINATE_NORM.keys():
            obj_id = meta_data.OBJECT_NAME2IDX[obj_name]
            xyz_min = meta_data.OBJECT_COORDINATE_NORM[obj_name]
            infos[obj_id] = {}
            infos[obj_id]['min_x'] = xyz_min[0]
            infos[obj_id]['min_y'] = xyz_min[1]
            infos[obj_id]['min_z'] = xyz_min[2]
        return infos


    def __len__(self, ):
        return len(self.frame_list)


    def __getitem__(self, idx):
        # this data loading process may be a bit heavy
        img_id = self.frame_list[idx]
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.dataset_metadata.parse_img_id(img_id)
        obj_id = self.dataset_metadata.OBJECT_NAME2IDX[obj_name]

        rgb = cv2.imread(self.dataset_metadata.get_image_path(img_id))
        # downsampling to speed up
        rgb = cv2.resize(rgb, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        _h, _w, _ = rgb.shape

        ########################################################################################################################
        # Here, we assume the mask of object is given which can be obtained from some segmentation models such as PointRend.
        object_full_mask = cv2.imread(self.dataset_metadata.get_object_full_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255
        if object_full_mask is None:
            object_full_mask = np.zeros((_h, _w))
        ########################################################################################################################

        cx, cy, fx, fy = self.dataset_metadata.cam_intrinsics[int(cam_id)]
        cam_K = np.array([[fx / 2, 0, cx / 2], [0, fy / 2, cy / 2], [0, 0, 1]])

        h, w = object_full_mask.shape
        ys, xs = np.nonzero(object_full_mask)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        box = np.array([x1, y1, x2 - x1, y2 - y1])

        c, s = self.xywh_to_cs(box, self.cfg.augment.pad_ratio, s_max=max(_h, _w))

        rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
        rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
        c = np.array([c_w_, c_h_]) * 2
        s = s_ * 2
        box = box * 2

        return img_id, obj_name, obj_id, rgb, c, s, np.asarray(box), cam_K.astype(np.float32)


class BEHAVEExtendObjectDataset(LM):

    def __init__(self, cfg, split='test'):
        assert split == 'test'
        self.cfg = cfg
        self.root_dir = cfg.dataset.root_dir
        self.split = split

        self.dataset_metadata = BEHAVEExtendMetaData(self.root_dir)
        self.infos = self.load_model_info()
        self.frame_list = list(self.dataset_metadata.go_through_all_frames(split=split))


    def load_model_info(self, ):
        infos = {}
        meta_data = self.dataset_metadata
        for obj_name in meta_data.OBJECT_COORDINATE_NORM.keys():
            obj_id = meta_data.OBJECT_NAME2IDX[obj_name]
            xyz_min = meta_data.OBJECT_COORDINATE_NORM[obj_name]
            infos[obj_id] = {}
            infos[obj_id]['min_x'] = xyz_min[0]
            infos[obj_id]['min_y'] = xyz_min[1]
            infos[obj_id]['min_z'] = xyz_min[2]
        return infos


    def __len__(self, ):
        return len(self.frame_list)


    def __getitem__(self, idx):
        # this data loading process may be a bit heavy
        img_id = self.frame_list[idx]
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.dataset_metadata.parse_img_id(img_id)
        obj_id = self.dataset_metadata.OBJECT_NAME2IDX[obj_name]

        try:
            rgb = cv2.imread(self.dataset_metadata.get_image_path(img_id))
            # downsampling to speed up
            rgb = cv2.resize(rgb, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            _h, _w, _ = rgb.shape
        except: # image may not exists
            return self.__getitem__(np.random.randint(len(self)))

        ########################################################################################################################
        # Here, we assume the mask of object is given which can be obtained from some segmentation models such as PointRend.
        object_full_mask = cv2.imread(self.dataset_metadata.get_object_full_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255
        if object_full_mask is None:
            object_full_mask = np.zeros((_h, _w))
        ########################################################################################################################

        cx, cy, fx, fy = self.dataset_metadata.cam_intrinsics[int(cam_id)]
        cam_K = np.array([[fx / 2, 0, cx / 2], [0, fy / 2, cy / 2], [0, 0, 1]])

        h, w = object_full_mask.shape
        ys, xs = np.nonzero(object_full_mask)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        box = np.array([x1, y1, x2 - x1, y2 - y1])

        c, s = self.xywh_to_cs(box, self.cfg.augment.pad_ratio, s_max=max(_h, _w))

        rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
        rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
        c = np.array([c_w_, c_h_]) * 2
        s = s_ * 2
        box = box * 2

        return img_id, obj_name, obj_id, rgb, c, s, np.asarray(box), cam_K.astype(np.float32)


class InterCapObjectDataset(LM):

    def __init__(self, cfg, split='test'):
        assert split == 'test'
        self.cfg = cfg
        self.root_dir = cfg.dataset.root_dir
        self.split = split

        self.dataset_metadata = InterCapMetaData(self.root_dir)
        self.infos = self.load_model_info()
        self.frame_list = list(self.dataset_metadata.go_through_all_frames(split=split))


    def load_model_info(self, ):
        infos = {}
        meta_data = self.dataset_metadata
        for obj_name in meta_data.OBJECT_COORDINATE_NORM.keys():
            obj_id = int(meta_data.OBJECT_NAME2IDX[obj_name]) - 1
            xyz_min = meta_data.OBJECT_COORDINATE_NORM[obj_name]
            infos[obj_id] = {}
            infos[obj_id]['min_x'] = xyz_min[0]
            infos[obj_id]['min_y'] = xyz_min[1]
            infos[obj_id]['min_z'] = xyz_min[2]
        return infos


    def __len__(self, ):
        return len(self.frame_list)


    def __getitem__(self, idx):
        # this data loading process may be a bit heavy
        img_id = self.frame_list[idx]
        sub_id, obj_id, seq_id, cam_id, frame_id = self.dataset_metadata.parse_img_id(img_id)
        obj_name = self.dataset_metadata.OBJECT_IDX2NAME[obj_id]
        obj_id = int(self.dataset_metadata.OBJECT_NAME2IDX[obj_name]) - 1

        rgb = cv2.imread(self.dataset_metadata.get_image_path(img_id))
        # downsampling to speed up
        rgb = cv2.resize(rgb, (0, 0),  fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        _h, _w, _ = rgb.shape

        ########################################################################################################################
        # Here, we assume the mask of object is given which can be obtained from some segmentation models such as PointRend.
        object_full_mask = cv2.imread(self.dataset_metadata.get_object_full_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255
        if object_full_mask is None:
            object_full_mask = np.zeros((_h, _w))
        ########################################################################################################################

        calitration = self.dataset_metadata.cam_calibration[cam_id]
        cx, cy = calitration['c']
        fx, fy = calitration['f']
        cam_K = np.array([[fx / 2, 0, cx / 2], [0, fy / 2, cy / 2], [0, 0, 1]])

        h, w = object_full_mask.shape
        ys, xs = np.nonzero(object_full_mask)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        box = np.array([x1, y1, x2 - x1, y2 - y1])

        c, s = self.xywh_to_cs(box, self.cfg.augment.pad_ratio, s_max=max(_h, _w))

        rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
        rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
        c = np.array([c_w_, c_h_]) * 2
        s = s_ * 2
        box = box * 2

        return img_id, obj_name, obj_id, rgb, c, s, np.asarray(box), cam_K.astype(np.float32)


def update_config(cfg, args):
    cfg.pytorch.exp_id = 'epropnp_behave' if args.is_behave else 'epropnp_intercap'
    cfg.pytorch.task = 'rot'
    cfg.pytorch.threads_num = 8
    cfg.pytorch.save_mode = 'last'
    file_dir = os.path.dirname(__file__)
    if args.is_behave:
        cfg.pytorch.save_path = os.path.join(file_dir, '../outputs/epropnp/behave')
        cfg.pytorch.load_model = os.path.join(file_dir, '../outputs/epropnp/behave/model_last.checkpoint')
    elif args.behave_extend:
        cfg.pytorch.save_path = os.path.join(file_dir, '../outputs/epropnp/behave_extend')
        cfg.pytorch.load_model = os.path.join(file_dir, '../outputs/epropnp/behave_extend/model_last.checkpoint')
    else:
        cfg.pytorch.save_path = os.path.join(file_dir, '../outputs/epropnp/intercap')
        cfg.pytorch.load_model = os.path.join(file_dir, '../outputs/epropnp/intercap/model_last.checkpoint')
    cfg.pytorch.gpu = 0

    cfg.pytorch['tensorboard'] = os.path.join(cfg.pytorch.save_path, 'tensorboard')
    if not os.path.exists(cfg.pytorch.tensorboard):
        os.makedirs(cfg.pytorch.tensorboard)
    cfg.writer = SummaryWriter(cfg.pytorch.tensorboard)

    cfg.dataset.root_dir = args.root_dir

    cfg.network.numBackLayers = 50
    cfg.network.rot_output_channels = 5

    cfg.train.begin_epoch = 0
    cfg.train.end_epoch = 40
    cfg.train.train_batch_size = 32
    cfg.train.lr_epoch_step = [30, 35, 40]
    cfg.test.disp_interval = 100


def main(args):
    cfg = get_base_config()
    update_config(cfg, args)
    model, _ = build_model(cfg)

    if args.is_behave:
        dataset = BEHAVEObjectDataset(cfg, 'test')
    elif args.behave_extend:
        dataset = BEHAVEExtendObjectDataset(cfg, 'test')
    else:
        dataset = InterCapObjectDataset(cfg, 'test')
    obj_info = dataset.load_model_info()

    def _worker_init_fn():
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=int(cfg.pytorch.threads_num),
        worker_init_fn=_worker_init_fn()
    )
    model.eval()
    model = model.cuda(cfg.pytorch.gpu)

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

    dataset_metadata = dataset.dataset_metadata
    for i, (img_ids, obj_name, obj_id, rgb, c_box, s_box, box, cam_K) in tqdm(enumerate(data_loader), desc='inference EPro-PnP', total=len(data_loader)):
        if cfg.pytorch.gpu > -1:
            rgb_var = rgb.cuda(cfg.pytorch.gpu).float()
            c_box_var = c_box.cuda(cfg.pytorch.gpu).float()
            s_box_var = s_box.cuda(cfg.pytorch.gpu).float()
            cam_K = cam_K.cuda(cfg.pytorch.gpu).float()
        else:
            rgb_var = rgb.float()
            c_box_var = c_box.float()
            s_box_var = s_box.float()
            cam_K = cam_K.float()

        bs = len(rgb_var)
        (noc, w2d, scale), pred_trans = model(rgb_var)

        dim = [[abs(obj_info[obj_id_]['min_x']),
                abs(obj_info[obj_id_]['min_y']),
                abs(obj_info[obj_id_]['min_z'])] for obj_id_ in obj_id.cpu().numpy()]
        dim = noc.new_tensor(dim) # (n, 3)
        x3d = noc.permute(0, 2, 3, 1) * dim[:, None, None, :] # (bs, h, w, 3)
        w2d = w2d.permute(0, 2, 3, 1) # (bs, h, w, 2)

        s = s_box_var.to(torch.int64)
        wh_begin = c_box_var.to(torch.int64) - s[:, None] / 2. # (n, 2)
        wh_unit = s.to(torch.float32) / cfg.dataiter.out_res  # (n, )

        wh_arange = torch.arange(cfg.dataiter.out_res, device=noc.device, dtype=torch.float32)
        y, x = torch.meshgrid(wh_arange, wh_arange)  # (h, w)
        # (bs, 2, h, w)
        x2d = torch.stack((wh_begin[:, 0, None, None] + x * wh_unit[:, None, None],
                           wh_begin[:, 1, None, None] + y * wh_unit[:, None, None]), dim=1)
        x2d = x2d.permute(0, 2, 3, 1)

        dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # Assuming no lens distortion

        # for fair comparison we use EPnP initialization
        pred_conf_np = w2d.mean(dim=-1).detach().cpu().numpy()  # (bs, h, w)
        binary_mask = pred_conf_np >= np.quantile(pred_conf_np.reshape(bs, -1), 0.8,
                                                  axis=1, keepdims=True)[..., None]
        R_quats = []
        T_vectors = []
        x2d_np = x2d.detach().cpu().numpy()
        x3d_np = x3d.detach().cpu().numpy()
        for x2d_np_, x3d_np_, mask_np_, K in zip(x2d_np, x3d_np, binary_mask, cam_K.cpu().numpy()):
            _, R_vector, T_vector = cv2.solvePnP(
                x3d_np_[mask_np_], x2d_np_[mask_np_], K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
            q = Rotation.from_rotvec(R_vector.reshape(-1)).as_quat()[[3, 0, 1, 2]]
            R_quats.append(q)
            T_vectors.append(T_vector.reshape(-1))
        R_quats = x2d.new_tensor(R_quats)
        T_vectors = x2d.new_tensor(T_vectors)
        pose_init = torch.cat((T_vectors, R_quats), dim=-1)  # (bs, 7)

        x2d = x2d.reshape(bs, -1, 2)
        w2d = w2d.reshape(bs, -1, 2)
        x3d = x3d.reshape(bs, -1, 3)
        w2d = w2d.sigmoid() * scale[:, None, :]
        camera = PerspectiveCamera(cam_mats=cam_K, z_min=0.01)
        cost_fun = AdaptiveHuberPnPCost(relative_delta=0.1)
        cost_fun.set_param(x2d, w2d)
        pose_opt = epropnp(x3d, x2d, w2d, camera, cost_fun, pose_init=pose_init, fast_mode=True)[0]

        for batch_idx, img_id in enumerate(img_ids):
            inference_result = {
                'x3d': x3d[batch_idx].detach().cpu().numpy(),
                'x2d': x2d[batch_idx].detach().cpu().numpy(),
                'w2d': w2d[batch_idx].detach().cpu().numpy(),
                'pose_opt': pose_opt[batch_idx].detach().cpu().numpy(),
            }
            out_path = dataset_metadata.get_pred_coor_map_path(img_id)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_pickle(inference_result, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str, help='Dataset root directory.')
    parser.add_argument('--is_behave', default=False, action='store_true', help='Process behave dataset or intercap dataset.')
    parser.add_argument('--behave_extend', default=False, action='store_true', help='Train on BEHAVE-Extended dataset.')
    args = parser.parse_args()

    main(args)
