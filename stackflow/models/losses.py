import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from smplx import SMPLHLayer, SMPLXLayer
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

from stackflow.models.ho_offset import HOOffset
from stackflow.utils.camera import perspective_projection
from stackflow.datasets.behave_metadata import BEHAVEMetaData
from stackflow.datasets.intercap_metadata import InterCapMetaData
from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData


class SMPLLoss(nn.Module):

    def __init__(self, cfg):
        super(SMPLLoss, self).__init__()
        self.cfg = cfg
        self.params_loss = nn.MSELoss(reduction='none')
        self.joint_loss = nn.L1Loss(reduction='none')

        if cfg.dataset.name == 'BEHAVE' or cfg.dataset.name == 'BEHAVE-Extended':
            self.smpl = SMPLHLayer(model_path=cfg.model.smplh_dir, gender='male')
        elif cfg.dataset.name == 'InterCap':
            self.smpl = SMPLXLayer(model_path=cfg.model.smplx_dir, gender='neutral')


    def forward(self, preds, targets):

        pred_cam = preds['pred_cam'] # [b, 3]
        b = pred_cam.shape[0]
        pred_betas = preds['pred_betas'] # [b, 10]
        pred_pose6d = preds['pred_theta'] # [b, 6 * (21 + 1)]
        pred_pose6d = pred_pose6d.reshape(b, -1, 6)
        pred_smpl_rotmat = rotation_6d_to_matrix(pred_pose6d)

        gt_smpl_betas = targets['smpl_betas']
        gt_smpl_rotmat = targets['smpl_pose_rotmat']

        loss_beta = self.params_loss(pred_betas, gt_smpl_betas).sum() / b
        loss_global_orient = self.params_loss(pred_smpl_rotmat[:, 0], gt_smpl_rotmat[:, 0]).sum() / b
        loss_body_pose = self.params_loss(pred_smpl_rotmat[:, 1:], gt_smpl_rotmat[:, 1:]).sum() / b
        pred_pose6d = pred_pose6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d = self.params_loss(torch.matmul(pred_pose6d.permute(0, 2, 1), pred_pose6d), 
            torch.eye(2, dtype=pred_pose6d.dtype, device=pred_pose6d.device).unsqueeze(0)).mean()

        gt_trans = targets['smpl_trans']
        box_size = targets['box_size'].reshape(b, )
        box_center = targets['box_center'].reshape(b, 2)
        optical_center = targets['optical_center'].reshape(b, 2)
        focal_length = targets['focal_length'].reshape(b, 2)

        x = pred_cam[:, 1] * box_size + box_center[:, 0] - optical_center[:, 0]
        y = pred_cam[:, 2] * box_size + box_center[:, 1] - optical_center[:, 1]
        z = focal_length[:, 0] / (box_size * pred_cam[:, 0] + 1e-9)
        x = x / focal_length[:, 0] * z
        y = y / focal_length[:, 1] * z
        pred_cam_t_global = torch.stack([x, y, z], dim=-1)
        loss_trans = F.l1_loss(pred_cam_t_global, gt_trans)

        smpl_pred_out = self.smpl(global_orient=pred_smpl_rotmat[:, 0:1], body_pose=pred_smpl_rotmat[:, 1:], betas=pred_betas)
        pred_joint_3d = smpl_pred_out.joints[:, :22]
        pred_joint_3d = pred_joint_3d - pred_joint_3d[:, :1]

        pred_cam_t_local = torch.stack([pred_cam[:, 1] / (pred_cam[:, 0] + 1e-9),
                                        pred_cam[:, 2] * focal_length[:, 0] / (focal_length[:, 1] * pred_cam[:, 0] + 1e-9),
                                        focal_length[:, 0] / (self.cfg.dataset.img_size * pred_cam[:, 0] + 1e-9 )], dim=-1)

        pred_joint_2d = perspective_projection(pred_joint_3d, trans=pred_cam_t_local, focal_length=focal_length / self.cfg.dataset.img_size)

        gt_joint_3d = targets['person_joint_3d']
        gt_joint_2d = targets['person_joint_2d']
        loss_joint_3d = self.joint_loss(pred_joint_3d - pred_joint_3d[:, 0:1], gt_joint_3d - gt_joint_3d[:, 0:1]).sum() / b
        loss_joint_2d = self.joint_loss(pred_joint_2d, gt_joint_2d).sum() / b

        losses = {
            'loss_beta': loss_beta,
            'loss_global_orient': loss_global_orient,
            'loss_body_pose': loss_body_pose,
            'loss_pose_6d': loss_pose_6d,
            'loss_trans': loss_trans,
            'loss_joint_3d': loss_joint_3d,
            'loss_joint_2d': loss_joint_2d,
        }

        return losses


class OffsetLoss(nn.Module):

    def __init__(self, cfg):
        super(OffsetLoss, self).__init__()
        self.cfg = cfg
        self.hooffset = HOOffset(cfg)
        self.object_loss = ObjectLoss(cfg)


    def forward(self, preds, targets):
        pred_gamma = preds['pred_gamma'] # [b, dim]
        b = pred_gamma.shape[0]

        object_labels = targets['object_labels']
        smpl_body_pose_rotmat = targets['smpl_pose_rotmat'][:, 1:]
        smpl_betas = targets['smpl_betas']
        object_rel_rotmat = targets['object_rel_rotmat']
        object_rel_trans = targets['object_rel_trans']

        gt_gamma, gt_offsets = self.hooffset.encode(smpl_betas, smpl_body_pose_rotmat, object_rel_rotmat, object_rel_trans, object_labels)

        loss_gamma = F.l1_loss(pred_gamma, gt_gamma)

        pred_betas = preds['pred_betas'] # [b, 10]
        pred_pose6d = preds['pred_theta'] # [b, 6 * (21 + 1)]
        pred_pose6d = pred_pose6d.reshape(b, -1, 6)
        pred_rotmat = rotation_6d_to_matrix(pred_pose6d)
        pred_global_pose_rotmat = pred_rotmat[:, 0]
        pred_smpl_body_rotmat = pred_rotmat[:, 1:]
        pred_offsets = self.hooffset.decode(pred_gamma, object_labels)
        pred_obj_rel_R, pred_obj_rel_T = self.hooffset.decode_object_RT(pred_offsets, pred_betas, pred_smpl_body_rotmat, object_labels)

        loss_offset = F.l1_loss(gt_offsets, pred_offsets)

        focal_length = targets['focal_length'].reshape(b, 2)
        pred_cam = preds['pred_cam'] # [b, 3]
        pred_cam_t_local = torch.stack([pred_cam[:, 1] / (pred_cam[:, 0] + 1e-9),
                                        pred_cam[:, 2] * focal_length[:, 0] / (focal_length[:, 1] * pred_cam[:, 0] + 1e-9),
                                        focal_length[:, 0] / (self.cfg.dataset.img_size * pred_cam[:, 0] + 1e-9)], dim=-1)

        pred_obj_R = torch.matmul(pred_global_pose_rotmat, pred_obj_rel_R)
        pred_obj_t = torch.matmul(pred_global_pose_rotmat, pred_obj_rel_T.unsqueeze(-1)).squeeze(-1) + pred_cam_t_local

        object_reproj_loss = self.object_loss(pred_obj_R, pred_obj_t, object_labels, targets['object_kpts_2d'], targets['object_kpts_weights'], focal_length / self.cfg.dataset.img_size)

        losses = {
            'loss_gamma': loss_gamma,
            'object_reproj_loss': object_reproj_loss,
            'loss_offset': loss_offset,
        }

        return losses


class ObjectLoss(nn.Module):

    def __init__(self, cfg):
        super(ObjectLoss, self).__init__()

        if cfg.dataset.name == 'BEHAVE':
            self.dataset_metadata = BEHAVEMetaData(cfg.dataset.root_dir)
        elif cfg.dataset.name == 'InterCap':
            self.dataset_metadata = InterCapMetaData(cfg.dataset.root_dir)
        elif cfg.dataset.name == 'BEHAVE-Extended':
            self.dataset_metadata = BEHAVEExtendMetaData(cfg.dataset.root_dir)

        num_object = len(self.dataset_metadata.OBJECT_IDX2NAME)
        object_keypoints = np.zeros((num_object, self.dataset_metadata.object_max_keypoint_num, 3))
        for idx, object_idx in enumerate(sorted(self.dataset_metadata.OBJECT_IDX2NAME.keys())):
            object_name = self.dataset_metadata.OBJECT_IDX2NAME[object_idx]
            keypoints = self.dataset_metadata.load_object_keypoints(object_name)
            object_keypoints[idx, :len(keypoints)] = keypoints
        object_keypoints = torch.tensor(object_keypoints, dtype=torch.float32)
        self.register_buffer('object_keypoints', object_keypoints)


    def forward(self, obj_R, obj_t, object_labels, gt_keypoints, keypoints_weights, focal_length):
        object_keypoints = self.object_keypoints[object_labels].clone()
        object_keypoints_2d = perspective_projection(object_keypoints, trans=obj_t, rotmat=obj_R, focal_length=focal_length)
        obj_reproj_loss = F.l1_loss(object_keypoints_2d, gt_keypoints, reduction='none')
        keypoints_weights = keypoints_weights.unsqueeze(-1).repeat(1, 1, 2)
        obj_reproj_loss = obj_reproj_loss * keypoints_weights
        obj_reproj_loss = obj_reproj_loss.sum() / keypoints_weights.sum()

        return obj_reproj_loss


class FlowLoss(nn.Module):

    def __init__(self, cfg, flow):
        super(FlowLoss, self).__init__()
        self.cfg = cfg
        self.flow = flow

        self.joint_loss = nn.L1Loss(reduction='none')
        self.params_loss = nn.MSELoss(reduction='none')

        if cfg.dataset.name == 'BEHAVE' or cfg.dataset.name == 'BEHAVE-Extended':
            self.smpl = SMPLHLayer(model_path=cfg.model.smplh_dir, gender='male')
        elif cfg.dataset.name == 'InterCap':
            self.smpl = SMPLXLayer(model_path=cfg.model.smplx_dir, gender='neutral')
        self.hooffset = HOOffset(cfg)
        self.object_loss = ObjectLoss(cfg)


    def forward(self, preds, targets):
        human_features = preds['human_features']
        hoi_features = preds['hoi_features']
        gt_smpl_rotmat = targets['smpl_pose_rotmat']
        b = gt_smpl_rotmat.shape[0]
        gt_pose_6d = matrix_to_rotation_6d(gt_smpl_rotmat).reshape(b, -1)

        object_labels = targets['object_labels']
        smpl_body_pose_rotmat = targets['smpl_pose_rotmat'][:, 1:]
        smpl_betas = targets['smpl_betas']
        object_rel_rotmat = targets['object_rel_rotmat']
        object_rel_trans = targets['object_rel_trans']

        gt_gamma, gt_offset = self.hooffset.encode(smpl_betas, smpl_body_pose_rotmat, object_rel_rotmat, object_rel_trans, object_labels)

        gt_pose_6d = gt_pose_6d + torch.randn_like(gt_pose_6d) * 0.001
        gt_gamma = gt_gamma + torch.randn_like(gt_gamma) * 0.01
        theta_log_prob, theta_z, gamma_log_prob, gamma_z = self.flow.log_prob(gt_pose_6d, gt_gamma, human_features, hoi_features, object_labels)
        loss_theta_nll = - theta_log_prob.mean()
        loss_gamma_nll = - gamma_log_prob.mean()

        num_samples = self.cfg.train.num_samples
        theta_samples, _, _, gamma_samples, _, _ = self.flow.sample(num_samples, human_features, hoi_features, object_labels)
        theta_samples = theta_samples.reshape(b * num_samples, -1)
        gamma_samples = gamma_samples.reshape(b * num_samples, -1)

        pred_betas = preds['pred_betas'].unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, -1) # [b * n, 10]
        pred_pose6d = theta_samples.reshape(b * num_samples, -1, 6)
        pred_smpl_rotmat = rotation_6d_to_matrix(pred_pose6d)
        smpl_pred_out = self.smpl(global_orient=pred_smpl_rotmat[:, 0:1], body_pose=pred_smpl_rotmat[:, 1:], betas=pred_betas)
        pred_joint_3d = smpl_pred_out.joints[:, :22]
        pred_joint_3d = pred_joint_3d - pred_joint_3d[:, :1]

        pred_pose6d = pred_pose6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d_samples = self.params_loss(torch.matmul(pred_pose6d.permute(0, 2, 1), pred_pose6d), 
            torch.eye(2, dtype=pred_pose6d.dtype, device=pred_pose6d.device).unsqueeze(0)).mean()

        pred_cam = preds['pred_cam'].unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, -1) # [b * n, 3]
        focal_length = targets['focal_length'].unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, 2) # [b * n, 2]
        pred_cam_t_local = torch.stack([pred_cam[:, 1] / (pred_cam[:, 0] + 1e-9),
                                        pred_cam[:, 2] * focal_length[:, 0] / (focal_length[:, 1] * pred_cam[:, 0] + 1e-9),
                                        focal_length[:, 0] / (self.cfg.dataset.img_size * pred_cam[:, 0] + 1e-9 )], dim=-1)
        pred_joint_2d = perspective_projection(pred_joint_3d, trans=pred_cam_t_local, focal_length=focal_length / self.cfg.dataset.img_size)

        gt_joint_2d = targets['person_joint_2d'].unsqueeze(1).repeat(1, num_samples, 1, 1)
        gt_joint_2d = gt_joint_2d.reshape(b * num_samples, -1, 2)
        loss_joint_2d = self.joint_loss(pred_joint_2d, gt_joint_2d).mean()

        object_labels = object_labels.unsqueeze(1).repeat(1, num_samples).reshape(b * num_samples, )
        pred_offsets = self.hooffset.decode(gamma_samples, object_labels)
        pred_obj_rel_R, pred_obj_rel_T = self.hooffset.decode_object_RT(pred_offsets, pred_betas, pred_smpl_rotmat[:, 1:], object_labels)

        pred_global_pose_rotmat = pred_smpl_rotmat[:, 0]
        pred_obj_R = torch.matmul(pred_global_pose_rotmat, pred_obj_rel_R)
        pred_obj_t = torch.matmul(pred_global_pose_rotmat, pred_obj_rel_T.unsqueeze(-1)).squeeze(-1) + pred_cam_t_local

        gt_object_keypoints = targets['object_kpts_2d'].unsqueeze(1).repeat(1, num_samples, 1, 1).reshape(b * num_samples, -1, 2)
        keypoints_weights = targets['object_kpts_weights'].unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, -1)

        object_reproj_loss = self.object_loss(pred_obj_R, pred_obj_t, object_labels, gt_object_keypoints, keypoints_weights, focal_length / self.cfg.dataset.img_size)

        losses = {
            'loss_theta_nll': loss_theta_nll,
            'loss_gamma_nll': loss_gamma_nll,
            'loss_joint_2d_sample': loss_joint_2d,
            'loss_pose_6d_samples': loss_pose_6d_samples,
            'object_sample_reproj_loss': object_reproj_loss,
        }

        return losses
