import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from smplx import SMPLHLayer, SMPLXLayer
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

from prohmr.models.backbones.resnet import resnet50
from prohmr.models.losses import Keypoint3DLoss, Keypoint2DLoss
from prohmr.utils.geometry import perspective_projection

from stackflow.models.hoi_flow import HOIFlow
from stackflow.models.rel_dist_decoder import DistanceDecoder

class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

        self.backbone = resnet50(pretrained=True)
        self.hoi_flow = HOIFlow(cfg)
        self.rel_dist_decoder = DistanceDecoder(cfg.data.pca_params_dir, 
                                                anchor_num=cfg.model.hoi_decoder.anchor_num,
                                                pca_dim=cfg.model.hoi_decoder.pca_dim)

        self.joint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.joint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smpl_parameter_loss = nn.MSELoss(reduction='none')

        if cfg.dataset.name == 'behave':
            self.smpl = SMPLHLayer(model_path=cfg.data.smplh_path, gender='male')
        else:
            self.smpl = SMPLXLayer(model_path=cfg.data.smplx_path, gender='male')
        self.loss_weights = {
            'loss_joint_2d_mode': 0.01,
            'loss_joint_2d_exp': 0.001,
            'loss_joint_3d_mode': 0.05,
            'loss_joint_3d_exp': 0.,
            'loss_smpl_beta_mode': 0.0005,
            'loss_smpl_beta_exp': 0.,
            'loss_smpl_body_pose_mode': 0.001,
            'loss_smpl_body_pose_exp': 0.,
            'loss_smpl_global_orient_mode': 0.001,
            'loss_smpl_global_orient_exp': 0.,
            'loss_pose_6d_mode': 0.1,
            'loss_pose_6d_exp': 0.1,
            'loss_trans': 0.,
            'loss_smpl_nll': 0.001,
            'loss_hoi_nll': 0.001,
            'loss_rel_dist_mode': 0, # 0.1,
            'loss_rel_dist_exp': 0.,
            'loss_lattent_codes_mode': 0.01,
            'loss_lattent_codes_exp': 0.,
        }
        self.optimizer = self.get_optimizer(cfg)


    def get_optimizer(self, cfg):
        optimizer = torch.optim.AdamW(params=list(self.backbone.parameters()) + list(self.hoi_flow.parameters()),
                                      lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        return optimizer


    def forward_step(self, batch):

        x = batch['image']
        batch_size = x.shape[0]

        conditioning_feats = self.backbone(x)

        object_labels = batch['object_label']
        if self.training:
            num_samples = 2
        else:
            num_samples = 2
        smpl_samples, hoi_samples, pred_cam = self.hoi_flow(conditioning_feats, object_labels, num_samples=num_samples-1)
        smpl_z0 = torch.zeros(batch_size, 1, (self.cfg.model.hoiflow.num_pose + 1) * 6, device=x.device, dtype=x.dtype)
        hoi_z0 = torch.zeros(batch_size, 1, self.cfg.model.hoi_decoder.pca_dim, device=x.device, dtype=x.dtype)
        smpl_samples_mode, hoi_samples_mode, pred_cam = self.hoi_flow(conditioning_feats, object_labels, z=(smpl_z0, hoi_z0))

        pred_pose_6d = smpl_samples['pose_6d']
        pred_pose_6d_mode = smpl_samples_mode['pose_6d']
        pred_pose_6d = torch.cat([pred_pose_6d_mode, pred_pose_6d], dim=1) # (b, num_samples, 22, 6)
        pred_pose_rotmat = smpl_samples['pose_rotmat']
        pred_pose_rotmat_mode = smpl_samples_mode['pose_rotmat']
        pred_pose_rotmat = torch.cat([pred_pose_rotmat_mode, pred_pose_rotmat], dim=1) # (b, num_samples, 22, 3, 3)
        pred_betas = smpl_samples['betas'] # (b, 10)

        hoi_lattent_codes = hoi_samples['hoi_lattent_codes'].reshape(batch_size, (num_samples - 1) * (num_samples - 1), -1)
        hoi_lattent_codes_mode = hoi_samples_mode['hoi_lattent_codes'].reshape(batch_size, 1, -1)
        hoi_lattent_codes = torch.cat([hoi_lattent_codes_mode, hoi_lattent_codes], dim=1) # (b, -1, 32)

        output = {}
        output['conditioning_feats'] = conditioning_feats
        focal_length = batch['focal_length'].reshape(batch_size, 1, 2).repeat(1, num_samples, 1)
        pred_cam = pred_cam.unsqueeze(1).repeat(1, num_samples, 1)
        output['pred_cam'] = pred_cam
        pred_cam_t = torch.stack([pred_cam[:, :, 1] / (pred_cam[:, :, 0] + 1e-9),
                                  pred_cam[:, :, 2] * focal_length[:, :, 0] / (focal_length[:, :, 1] * pred_cam[:, :, 0] + 1e-9),
                                  focal_length[:, :, 0] / (self.cfg.dataset.img_size * pred_cam[:, :, 0] + 1e-9 )], dim=-1)
        output['pred_cam_t'] = pred_cam_t

        output['pred_pose_6d'] = pred_pose_6d
        output['pred_pose_rotmat'] = pred_pose_rotmat
        output['pred_betas'] = pred_betas

        smpl_global_orient = pred_pose_rotmat[:, :, 0:1].reshape(batch_size * num_samples, 1, 3, 3)
        smpl_body_pose = pred_pose_rotmat[:, :, 1:].reshape(batch_size * num_samples, -1, 3, 3)
        smpl_betas = pred_betas.unsqueeze(1).repeat(1, num_samples, 1).reshape(batch_size * num_samples, -1)
        smpl_output = self.smpl(global_orient=smpl_global_orient, body_pose=smpl_body_pose, betas=smpl_betas, pose2rot=False)
        pred_joint_3d = smpl_output.joints
        output['pred_joint_3d'] = pred_joint_3d.reshape(batch_size, num_samples, -1, 3)

        pred_cam_t = pred_cam_t.reshape(batch_size * num_samples, 3)
        focal_length = focal_length.reshape(batch_size * num_samples, 2)
        pred_joint_2d = perspective_projection(pred_joint_3d, translation=pred_cam_t, focal_length=focal_length / self.cfg.dataset.img_size)
        output['pred_joint_2d'] = pred_joint_2d.reshape(batch_size, num_samples, -1, 2)

        output['hoi_lattent_codes'] = hoi_lattent_codes

        return output


    def compute_loss(self, batch, output):
        pred_pose_6d = output['pred_pose_6d'] # (b, n, 22, 6)
        pred_pose_rotmat = output['pred_pose_rotmat'] # (b, n, 22, 3, 3)
        pred_betas = output['pred_betas'] # (b, 10)
        pred_joint_2d = output['pred_joint_2d'][:, :, :22] # (b, n, 22, 2)
        pred_joint_3d = output['pred_joint_3d'][:, :, :22] # (b, n, 22, 3)
        pred_cam = output['pred_cam'] # (b, n, 3)
        conditioning_feats = output['conditioning_feats'] # (b, 2048)
        batch_size, num_samples, _, _ = pred_pose_6d.shape
        device, dtype = pred_pose_6d.device, pred_pose_6d.dtype

        gt_joint_2d = batch['joint_2d'][:, :22]
        gt_joint_3d = batch['joint_3d'][:, :22]
        gt_smpl_betas = batch['smpl_betas'] # (b, 10)
        gt_smpl_pose = batch['smpl_pose'] # (b, 66)
        gt_smpl_global_orient = batch['smpl_global_orient'] # (b, 3)
        gt_smpl_pose_rotmat = axis_angle_to_matrix(gt_smpl_pose.reshape(batch_size, -1, 3)).unsqueeze(1).repeat(1, num_samples, 1, 1, 1)

        loss_joint_2d = self.joint_2d_loss(pred_joint_2d, gt_joint_2d.unsqueeze(1).repeat(1, num_samples, 1, 1))
        loss_joint_3d = self.joint_3d_loss(pred_joint_3d, gt_joint_3d.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_id=0)
        loss_joint_2d_mode = loss_joint_2d[:, [0]].sum() / batch_size
        if loss_joint_2d.shape[1] > 1:
            loss_joint_2d_exp = loss_joint_2d[:, 1:].sum() / (batch_size * (num_samples - 1))
        else:
            loss_joint_2d_exp = 0
        loss_joint_3d_mode = loss_joint_3d[:, [0]].sum() / batch_size
        if loss_joint_3d.shape[1] > 1:
            loss_joint_3d_exp = loss_joint_3d[:, 1:].sum() / (batch_size * (num_samples - 1))
        else:
            loss_joint_3d_exp = 0

        loss_smpl_beta = self.smpl_parameter_loss(pred_betas, gt_smpl_betas)
        loss_smpl_body_pose = self.smpl_parameter_loss(pred_pose_rotmat[:, :, 1:22], gt_smpl_pose_rotmat[:, :, 1:22])
        loss_smpl_global_orient = self.smpl_parameter_loss(pred_pose_rotmat[:, :, 0], gt_smpl_pose_rotmat[:, :, 0])
        loss_smpl_beta_mode = loss_smpl_beta[:, [0]].sum() / batch_size
        if loss_smpl_beta.shape[1] > 1:
            loss_smpl_beta_exp = loss_smpl_beta[:, 1:].sum() / (batch_size * (num_samples - 1))
        loss_smpl_body_pose_mode = loss_smpl_body_pose[:, [0]].sum() / batch_size
        if loss_smpl_body_pose.shape[1] > 1:
            loss_smpl_body_pose_exp = loss_smpl_body_pose[:, 1:].sum() / (batch_size * (num_samples - 1))
        loss_smpl_global_orient_mode = loss_smpl_global_orient[:, [0]].sum() / batch_size
        if loss_smpl_global_orient.shape[1] > 1:
            loss_smpl_global_orient_exp = loss_smpl_global_orient[:, 1:].sum() / (batch_size * (num_samples - 1))

        pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2, device=pred_pose_6d.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2)
        loss_pose_6d = loss_pose_6d.reshape(batch_size, num_samples, -1)
        loss_pose_6d_mode = loss_pose_6d[:, 0].mean()
        loss_pose_6d_exp = loss_pose_6d[:, 1:].mean()

        gt_trans = batch['smpl_trans']
        box_size = batch['box_size'].reshape(batch_size, )
        box_center = batch['box_center'].reshape(batch_size, 2)
        optical_center = batch['optical_center'].reshape(batch_size, 2)
        focal_length = batch['focal_length'].reshape(batch_size, 2)
        pred_cam = output['pred_cam'][:, 0].reshape(batch_size, 3)
        x = pred_cam[:, 1] * box_size + box_center[:, 0] - optical_center[:, 0]
        y = pred_cam[:, 2] * box_size + box_center[:, 1] - optical_center[:, 1]
        z = focal_length[:, 0] / (box_size * pred_cam[:, 0] + 1e-9)
        x = x / focal_length[:, 0] * z
        y = y / focal_length[:, 1] * z
        pred_cam_t = torch.stack([x, y, z], dim=-1)
        loss_trans = F.l1_loss(pred_cam_t, gt_trans)

        hoi_lattent_codes = output['hoi_lattent_codes'] # (b, n, n, 32)
        hoi_lattent_codes = hoi_lattent_codes.reshape(batch_size * num_samples, -1)
        object_labels = batch['object_label'].reshape(batch_size, 1).repeat(1, num_samples).reshape(-1)
        rel_dist_recon = self.rel_dist_decoder.forward(hoi_lattent_codes, object_labels)
        rel_dist_gt = self.rel_dist_decoder.get_rel_dist(batch['smpl_v_orig'], batch['object_v_orig'], batch['object_label'])
        _rel_dist_gt = rel_dist_gt.reshape(batch_size, 1, -1).repeat(1, num_samples, 1)
        rel_dist_recon = rel_dist_recon.reshape(batch_size, num_samples, -1)
        loss_rel_dist = F.l1_loss(rel_dist_recon, _rel_dist_gt, reduction='none')
        loss_rel_dist_mode = loss_rel_dist[:, 0].mean()
        loss_rel_dist_exp = loss_rel_dist[:, 1:,].mean()

        gt_hoi_lattent_codes = self.rel_dist_decoder.encode(_rel_dist_gt.reshape(batch_size * num_samples, -1), object_labels)
        loss_lattent_codes = F.l1_loss(hoi_lattent_codes, gt_hoi_lattent_codes, reduction='none')
        loss_lattent_codes = loss_lattent_codes.reshape(batch_size, num_samples, -1)
        loss_lattent_codes_mode = loss_lattent_codes[:, 0].mean()
        loss_lattent_codes_exp = loss_lattent_codes[:, 1].mean()

        gt_lattent_codes = self.rel_dist_decoder.encode(rel_dist_gt, batch['object_label'])
        gt_smpl_pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(gt_smpl_pose.reshape(batch_size, -1, 3))).reshape(batch_size, -1)
        if self.training:
            gt_lattent_codes = gt_lattent_codes + 0.01 * torch.randn_like(gt_lattent_codes)
            gt_smpl_pose_6d = gt_smpl_pose_6d + 0.001 * torch.randn_like(gt_smpl_pose_6d)
        smpl_log_prob, _, hoi_log_prob, _ = self.hoi_flow.log_prob(gt_smpl_pose_6d, gt_smpl_betas, gt_lattent_codes, conditioning_feats, batch['object_label'])
        loss_smpl_nll = - smpl_log_prob.mean()
        loss_hoi_nll = - hoi_log_prob.mean()

        loss = self.loss_weights['loss_joint_2d_mode'] * loss_joint_2d_mode + \
               self.loss_weights['loss_joint_2d_exp'] * loss_joint_2d_exp + \
               self.loss_weights['loss_joint_3d_mode'] * loss_joint_3d_mode + \
               self.loss_weights['loss_joint_3d_exp'] * loss_joint_3d_exp + \
               self.loss_weights['loss_smpl_beta_mode'] * loss_smpl_beta_mode + \
               self.loss_weights['loss_smpl_beta_exp'] * loss_smpl_beta_exp + \
               self.loss_weights['loss_smpl_body_pose_mode'] * loss_smpl_body_pose_mode + \
               self.loss_weights['loss_smpl_body_pose_exp'] * loss_smpl_body_pose_exp + \
               self.loss_weights['loss_smpl_global_orient_mode'] * loss_smpl_global_orient_mode + \
               self.loss_weights['loss_smpl_global_orient_exp'] * loss_smpl_global_orient_exp + \
               self.loss_weights['loss_pose_6d_mode'] * loss_pose_6d_mode + \
               self.loss_weights['loss_pose_6d_exp'] * loss_pose_6d_exp + \
               self.loss_weights['loss_trans'] * loss_trans + \
               self.loss_weights['loss_smpl_nll'] * loss_smpl_nll + \
               self.loss_weights['loss_hoi_nll'] * loss_hoi_nll + \
               self.loss_weights['loss_rel_dist_mode'] * loss_rel_dist_mode + \
               self.loss_weights['loss_rel_dist_exp'] * loss_rel_dist_exp + \
               self.loss_weights['loss_lattent_codes_mode'] * loss_lattent_codes_mode + \
               self.loss_weights['loss_lattent_codes_exp'] * loss_lattent_codes_exp

        losses = dict(loss=loss.detach(),
                      loss_joint_2d_mode=loss_joint_2d_mode.detach(),
                      loss_joint_2d_exp=loss_joint_2d_exp.detach(),
                      loss_joint_3d_mode=loss_joint_3d_mode.detach(),
                      loss_joint_3d_exp=loss_joint_3d_exp.detach(),
                      loss_smpl_beta_mode=loss_smpl_beta_mode.detach(),
                      loss_smpl_beta_exp=loss_smpl_beta_exp.detach(),
                      loss_smpl_body_pose_mode=loss_smpl_body_pose_mode.detach(),
                      loss_smpl_body_pose_exp=loss_smpl_body_pose_exp.detach(),
                      loss_smpl_global_orient_mode=loss_smpl_global_orient_mode.detach(),
                      loss_smpl_global_orient_exp=loss_smpl_global_orient_exp.detach(),
                      loss_pose_6d_mode=loss_pose_6d_mode.detach(),
                      loss_pose_6d_exp=loss_pose_6d_exp.detach(),
                      loss_trans=loss_trans.detach(),
                      loss_smpl_nll=loss_smpl_nll.detach(),
                      loss_hoi_nll=loss_hoi_nll.detach(),
                      loss_rel_dist_mode=loss_rel_dist_mode.detach(),
                      loss_rel_dist_exp=loss_rel_dist_exp.detach(),
                      loss_lattent_codes_mode=loss_lattent_codes_mode.detach(),
                      loss_lattent_codes_exp=loss_lattent_codes_exp.detach(),
                      )
        output['losses'] = losses

        return loss


    def train_step(self, batch):
        output = self.forward_step(batch)
        loss = self.compute_loss(batch, output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return output


    def validation_step(self, batch):
        with torch.no_grad():
            output = self.forward_step(batch)
            loss = self.compute_loss(batch, output)
        return output


    def inference_step(self, batch):
        with torch.no_grad():
            output = self.forward_step(batch)
            
            pred_pose_6d = output['pred_pose_6d']
            batch_size, num_samples, _, _ = pred_pose_6d.shape
            device, dtype = pred_pose_6d.device, pred_pose_6d.dtype

            box_size = batch['box_size'].reshape(batch_size, )
            box_center = batch['box_center'].reshape(batch_size, 2)
            image_center = batch['optical_center'].reshape(batch_size, 2)
            focal_length = batch['focal_length'].reshape(batch_size, 2)
            pred_cam = output['pred_cam'][:, 0].reshape(batch_size, 3)
            x = pred_cam[:, 1] * box_size + box_center[:, 0] - image_center[:, 0]
            y = pred_cam[:, 2] * box_size + box_center[:, 1] - image_center[:, 1]
            z = focal_length[:, 0] / (box_size * pred_cam[:, 0] + 1e-9)
            x = x / focal_length[:, 0] * z
            y = y / focal_length[:, 1] * z
            pred_cam_t = torch.stack([x, y, z], dim=-1)
            output['translation'] = pred_cam_t

            hoi_lattent_codes = output['hoi_lattent_codes'].reshape(batch_size * ((num_samples - 1) * (num_samples - 1) + 1), -1)
            object_labels = batch['object_label'].unsqueeze(-1).repeat(1, (num_samples - 1) * (num_samples - 1) + 1).reshape(-1)
            rel_dist_recon = self.rel_dist_decoder(hoi_lattent_codes, object_labels)
            rel_dist_recon = rel_dist_recon.reshape(batch_size, (num_samples - 1) * (num_samples - 1) + 1, -1)
            output['rel_dist_recon'] = rel_dist_recon

        return output


    def save_checkpoint(self, epoch, dir_, name):
        torch.save({
            'epoch': epoch,
            'backbone': self.backbone.state_dict(),
            'hoi_flow': self.hoi_flow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }, os.path.join(dir_, name))


    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.backbone.load_state_dict(state_dict['backbone'])
        self.hoi_flow.load_state_dict(state_dict['hoi_flow'])
        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        except:
            pass

        return state_dict['epoch']
