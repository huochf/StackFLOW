import torch
import torch.nn as nn

from pytorch3d.transforms import rotation_6d_to_matrix

from .backbone import build_backbone
from .cam_header import FCHeader
from .stackflow import StackFlow
from .losses import SMPLLoss, OffsetLoss, FlowLoss
from stackflow.utils.camera import perspective_projection


class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.header = FCHeader(cfg)
        self.stackflow = StackFlow(cfg)

        self.optimizer = torch.optim.AdamW(params=list(self.backbone.parameters()) + list(self.header.parameters()) + list(self.stackflow.parameters()),
                                           lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        # Encapsulating loss into three following parts improves the program readability, but at the cost of efficiency.
        self.smpl_loss = SMPLLoss(cfg)
        self.offset_loss = OffsetLoss(cfg)
        self.flow_loss = FlowLoss(cfg, self.stackflow)

        self.loss_weights = {
            'loss_beta': 0.005,
            'loss_global_orient': 0.01,
            'loss_body_pose': 0.05,
            'loss_pose_6d': 0.1,
            'loss_pose_6d_samples': 0.1,
            'loss_trans': 0.1,
            'loss_joint_3d': 0.05,
            'loss_joint_2d': 0.01,
            'loss_gamma': 0.1,
            'loss_offset': 0.01,
            'object_reproj_loss': 0.01,
            'loss_theta_nll': 0.001,
            'loss_gamma_nll': 0.001,
            'loss_joint_2d_sample': 0.01,
            'object_sample_reproj_loss': 0.01,
        }


    def forward(self, batch):
        image = batch['image']
        batch_size = image.shape[0]
        if self.cfg.model.backbone == 'resnet':
            visual_features = self.backbone(image)
            visual_features = visual_features.reshape(batch_size, -1)
            global_features = human_features = hoi_features = visual_features
        elif self.cfg.model.backbone == 'ViT':
            global_features, human_features, hoi_features = self.backbone(image)

        pred_betas, pred_cam = self.header(global_features)
        pred_theta, theta_log_prob, theta_z, pred_gamma, gamma_log_prob, gamma_z = self.stackflow(
            human_features=human_features, hoi_features=hoi_features, object_labels=batch['object_labels'])
        out = {
            'human_features': human_features,
            'hoi_features': hoi_features,
            'pred_betas': pred_betas,
            'pred_cam': pred_cam,
            'pred_theta': pred_theta,
            'theta_log_prob': theta_log_prob,
            'theta_z': theta_z,
            'pred_gamma': pred_gamma,
            'gamma_log_prob': gamma_log_prob,
            'gamma_z': gamma_z,
        }
        return out


    def forward_train(self, batch):
        pred = self.forward(batch)

        all_losses = {}
        loss_smpl = self.smpl_loss(pred, batch)
        loss_offset = self.offset_loss(pred, batch)
        loss_flow = self.flow_loss(pred, batch)
        all_losses.update(loss_smpl)
        all_losses.update(loss_offset)
        all_losses.update(loss_flow)
        loss = sum([v * self.loss_weights[k] for k, v in all_losses.items()])
        return loss, all_losses


    def inference(self, batch, debug=False):
        pred = self.forward(batch)

        pred_betas = pred['pred_betas']
        pred_pose6d = pred['pred_theta'] # [b, 6 * (21 + 1)]
        b = pred_pose6d.shape[0]
        pred_pose6d = pred_pose6d.reshape(b, -1, 6)
        pred_smpl_rotmat = rotation_6d_to_matrix(pred_pose6d)

        box_size = batch['box_size'].reshape(b, )
        box_center = batch['box_center'].reshape(b, 2)
        optical_center = batch['optical_center'].reshape(b, 2)
        focal_length = batch['focal_length'].reshape(b, 2)

        pred_cam = pred['pred_cam']
        x = pred_cam[:, 1] * box_size + box_center[:, 0] - optical_center[:, 0]
        y = pred_cam[:, 2] * box_size + box_center[:, 1] - optical_center[:, 1]
        z = focal_length[:, 0] / (box_size * pred_cam[:, 0] + 1e-9)
        x = x / focal_length[:, 0] * z
        y = y / focal_length[:, 1] * z
        hoi_trans = torch.stack([x, y, z], dim=-1) # [b, 3]
        hoi_rotmat = pred_smpl_rotmat[:, 0] # [b, 3, 3]

        pred_gamma = pred['pred_gamma'] # [b, dim]
        object_labels = batch['object_labels']
        pred_offsets = self.offset_loss.hooffset.decode(pred_gamma, object_labels)
        pred_obj_rel_R, pred_obj_rel_T = self.offset_loss.hooffset.decode_object_RT(pred_offsets, pred_betas, pred_smpl_rotmat[:, 1:], object_labels)

        # _pred_offsets, _pred_offsets = self.offset_loss.hooffset.encode(pred_betas, pred_smpl_rotmat[:, 1:], pred_obj_rel_R, pred_obj_rel_T, object_labels)

        results = {
            'pred_betas': pred_betas, # [b, 10]
            'pred_pose6d': pred_pose6d, # [b, 22, 6]
            'pred_smpl_body_pose': pred_smpl_rotmat[:, 1:], # [b, 21, 3, 3]
            'hoi_trans': hoi_trans, # [b, 3]
            'hoi_rotmat': hoi_rotmat, # [b, 3, 3]
            'pred_offsets': pred_offsets,
            # 'pred_offsets': _pred_offsets,
            'pred_obj_rel_R': pred_obj_rel_R,
            'pred_obj_rel_T': pred_obj_rel_T,

            'human_features': pred['human_features'], # for post-optimization
            'hoi_features': pred['hoi_features'], # for post-optimization
        }

        if debug:
            smpl_pred_out = self.smpl_loss.smpl(global_orient=pred_smpl_rotmat[:, 0:1], body_pose=pred_smpl_rotmat[:, 1:], betas=pred_betas)
            pred_joint_3d = smpl_pred_out.joints[:, :22]
            pred_joint_3d = pred_joint_3d - pred_joint_3d[:, :1]

            pred_cam_t_local = torch.stack([pred_cam[:, 1] / (pred_cam[:, 0] + 1e-9),
                                            pred_cam[:, 2] * focal_length[:, 0] / (focal_length[:, 1] * pred_cam[:, 0] + 1e-9),
                                            focal_length[:, 0] / (self.cfg.dataset.img_size * pred_cam[:, 0] + 1e-9 )], dim=-1)

            pred_joint_2d = perspective_projection(pred_joint_3d, trans=pred_cam_t_local, focal_length=focal_length / self.cfg.dataset.img_size)

            pred_obj_rel_R = batch['object_rel_rotmat']
            pred_obj_rel_T = batch['object_rel_trans']
            pred_obj_R = torch.matmul(hoi_rotmat, pred_obj_rel_R)
            pred_obj_t = torch.matmul(hoi_rotmat, pred_obj_rel_T.unsqueeze(-1)).squeeze(-1) + pred_cam_t_local

            object_keypoints = self.offset_loss.object_loss.object_keypoints[object_labels]
            object_keypoints_2d = perspective_projection(object_keypoints, trans=pred_obj_t, rotmat=pred_obj_R, focal_length=focal_length / self.cfg.dataset.img_size)

            results['pred_joint_2d'] = pred_joint_2d
            results['object_keypoints_2d'] = object_keypoints_2d

            num_samples = 4
            human_features = pred['human_features']
            hoi_features = pred['hoi_features']
            theta_samples, _, _, gamma_samples, _, _ = self.flow_loss.flow.sample(num_samples, human_features, hoi_features, object_labels)

            object_labels = object_labels.unsqueeze(1).repeat(1, num_samples).reshape(b * num_samples)
            pred_cam_t_local = pred_cam_t_local.unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, 3)
            focal_length = focal_length.unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, 2)
            pred_betas = pred_betas.unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, 10)
            hoi_rotmat = hoi_rotmat.unsqueeze(1).repeat(1, num_samples, 1, 1).reshape(b * num_samples, 3, 3)

            pred_pose6d = theta_samples.reshape(b * num_samples, -1, 6)
            pred_smpl_rotmat = rotation_6d_to_matrix(pred_pose6d)
            smpl_pred_out = self.smpl_loss.smpl(global_orient=pred_smpl_rotmat[:, 0:1], body_pose=pred_smpl_rotmat[:, 1:], betas=pred_betas)
            pred_joint_3d = smpl_pred_out.joints[:, :22]
            pred_joint_3d = pred_joint_3d - pred_joint_3d[:, :1]
            pred_joint_2d = perspective_projection(pred_joint_3d, trans=pred_cam_t_local, focal_length=focal_length / self.cfg.dataset.img_size)

            gamma_samples = gamma_samples.reshape(b * num_samples, -1)
            pred_offsets = self.offset_loss.hooffset.decode(gamma_samples, object_labels)
            pred_obj_rel_R, pred_obj_rel_T = self.offset_loss.hooffset.decode_object_RT(pred_offsets, pred_betas, pred_smpl_rotmat[:, 1:], object_labels)
        
            pred_obj_R = torch.matmul(hoi_rotmat, pred_obj_rel_R)
            pred_obj_t = torch.matmul(hoi_rotmat, pred_obj_rel_T.unsqueeze(-1)).squeeze(-1) + pred_cam_t_local

            object_keypoints = self.offset_loss.object_loss.object_keypoints[object_labels]
            object_keypoints_2d = perspective_projection(object_keypoints, trans=pred_obj_t, rotmat=pred_obj_R, focal_length=focal_length / self.cfg.dataset.img_size)

            results['pred_joint_2d_samples'] = pred_joint_2d.reshape(b, num_samples, -1, 2)
            results['object_keypoints_2d_samples'] = object_keypoints_2d.reshape(b, num_samples, -1, 2)

        return results


    def train_step(self, batch):
        self.optimizer.zero_grad()
        loss, all_losses = self.forward_train(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=10, norm_type=2)
        self.optimizer.step()

        return loss, all_losses


    def save_checkpoint(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'backbone': self.backbone.state_dict(),
            'header': self.header.state_dict(),
            'stackflow': self.stackflow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)


    def load_checkpoint(self, path):
        state_dict = torch.load(path)
        self.backbone.load_state_dict(state_dict['backbone'])
        self.header.load_state_dict(state_dict['header'])
        self.stackflow.load_state_dict(state_dict['stackflow'])
        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        except:
            print('Warning: Lacking weights for optimizer.')
        return state_dict['epoch']
