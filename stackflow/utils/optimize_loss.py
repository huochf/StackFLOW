import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from prohmr.utils.geometry import perspective_projection
from stackflow.utils.body_prior import create_prior
from stackflow.datasets.behave_metadata import SMPLH_OPENPOSE_INDICES


class RelativeDistanceLoss(nn.Module):

    def __init__(self, rel_dist, smpl_anchor_indices, object_anchor_indices, ):
        super(RelativeDistanceLoss, self).__init__()
        self.register_buffer('rel_dist', rel_dist)
        self.register_buffer('smpl_anchor_indices', smpl_anchor_indices)
        self.register_buffer('object_anchor_indices', object_anchor_indices)


    def forward(self, hoi_dict):
        smpl_v = hoi_dict['smpl_v_orig'][:, self.smpl_anchor_indices]
        object_v = torch.stack([
            hoi_dict['object_v_orig'][idx, indices]
            for idx, indices in enumerate(self.object_anchor_indices)
        ], dim=0)

        batch_size = smpl_v.shape[0]
        smpl_v = smpl_v.reshape(batch_size, -1, 1, 3)
        object_v = object_v.reshape(batch_size, 1, -1, 3)
        rel_dist = object_v - smpl_v
        rel_dist = rel_dist.reshape(batch_size, -1)
        rel_dist_loss = F.l1_loss(rel_dist, self.rel_dist, reduction='mean')

        return {'rel_dist_loss': rel_dist_loss}


class RelativeDistanceFlowLoss(nn.Module):

    def __init__(self, rel_dist_decoder, smpl_anchor_indices, object_anchor_indices, object_labels):
        super(RelativeDistanceFlowLoss, self).__init__()
        self.rel_dist_decoder = rel_dist_decoder
        self.register_buffer('smpl_anchor_indices', smpl_anchor_indices)
        self.register_buffer('object_anchor_indices', object_anchor_indices)
        self.register_buffer('object_labels', object_labels)


    def forward(self, hoi_dict):
        lattent_codes = hoi_dict['hoi_lattent_codes']
        rel_dist_flow = self.rel_dist_decoder(lattent_codes, self.object_labels)
        smpl_v = hoi_dict['smpl_v_orig'][:, self.smpl_anchor_indices]
        object_v = torch.stack([
            hoi_dict['object_v_orig'][idx, indices]
            for idx, indices in enumerate(self.object_anchor_indices)
        ], dim=0)

        batch_size = smpl_v.shape[0]
        smpl_v = smpl_v.reshape(batch_size, -1, 1, 3)
        object_v = object_v.reshape(batch_size, 1, -1, 3)
        rel_dist = object_v - smpl_v
        rel_dist = rel_dist.reshape(batch_size, -1)
        rel_dist_loss = F.l1_loss(rel_dist, rel_dist_flow, reduction='mean')

        return {'rel_dist_loss': rel_dist_loss}


class SMPLPriorLoss(nn.Module):

    def __init__(self, ):
        super(SMPLPriorLoss, self).__init__()
        self.betas_prior = create_prior(prior_type='l2')
        self.pose_prior = create_prior(prior_type='l2')


    def forward(self, hoi_dict):
        betas = hoi_dict['smpl_betas']
        body_pose = hoi_dict['smpl_body_pose']
        beta_prior_loss = self.betas_prior(betas)
        pose_prior_l2_loss = self.pose_prior(body_pose)

        return {
            'beta_prior_loss': beta_prior_loss,
            'pose_prior_l2_loss': pose_prior_l2_loss,
        }


class PersonKeypointLoss(nn.Module):

    def __init__(self, keypoints, confidence, K, smpl_type='smplh'):
        super(PersonKeypointLoss, self).__init__()
        batch_size = keypoints.shape[0]
        self.register_buffer('keypoints', keypoints.reshape(batch_size, -1, 2))
        self.register_buffer('confidence', confidence.reshape(batch_size, -1, 1))
        self.confidence[:, [8, 11]] *= 0.
        self.confidence[self.keypoints[:, :, 0] < 0] = 0
        self.register_buffer('K', K.reshape(batch_size, 3, 3).to(torch.float32))
        if smpl_type == 'smplh':
            self.register_buffer('indices', torch.tensor([52, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 53, 54, 55, 56], dtype=torch.int64))
        else:
            self.register_buffer('indices', torch.tensor([55, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59], dtype=torch.int64))


    def forward(self, hoi_dict):
        smpl_J = hoi_dict['smpl_J']
        batch_size = smpl_J.shape[0]
        focal_length = torch.stack([self.K[:, 0, 0], self.K[:, 1, 1]], dim=1).to(torch.float32)
        camera_center = self.K[:, :-1, -1]
        T = torch.zeros(batch_size, 3, dtype=torch.float32, device=smpl_J.device)
        R = torch.eye(3, dtype=torch.float32, device=smpl_J.device).reshape(1, 3, 3).repeat(batch_size, 1, 1)
        joints_reproj = perspective_projection(smpl_J, T, focal_length, camera_center, R)

        loss = F.l1_loss(joints_reproj[:, self.indices], self.keypoints, reduction='none') * self.confidence
        loss = loss.mean()

        return {
            'person_keypoints_reproj_loss': loss,
        }


class ObjectReprojLoss(nn.Module):

    def __init__(self, model_points, image_points, pts_confidence, K, rescaled=False):
        super(ObjectReprojLoss, self).__init__()
        batch_size = model_points.shape[0]
        self.register_buffer('model_points', model_points.reshape(batch_size, -1, 3).to(torch.float32))
        self.register_buffer('image_points', image_points.reshape(batch_size, -1, 2).to(torch.float32))
        self.register_buffer('pts_confidence', pts_confidence.reshape(batch_size, -1, 2).to(torch.float32))
        self.register_buffer('K', K.reshape(batch_size, 3, 3).to(torch.float32))
        self.rescaled = rescaled


    def forward(self, hoi_dict):
        batch_size = self.model_points.shape[0]
        object_R = hoi_dict['object_global_rotmat'].reshape(batch_size, 3, 3).to(torch.float32)
        object_T = hoi_dict['object_global_trans'].reshape(batch_size, 3).to(torch.float32)
        focal_length = torch.stack([self.K[:, 0, 0], self.K[:, 1, 1]], dim=1).to(torch.float32)
        camera_center = self.K[:, :-1, -1]
        object_reproj = perspective_projection(self.model_points, object_T, focal_length, camera_center, object_R)

        loss_obj_reproj = F.l1_loss(self.image_points, object_reproj, reduction='none') * self.pts_confidence
        if self.rescaled:
            loss_obj_reproj = loss_obj_reproj * ((self.pts_confidence.reshape(batch_size, -1).sum(-1).reshape(batch_size, 1, 1) - 15) / 10).exp()
            # loss_obj_reproj = loss_obj_reproj * ((self.pts_confidence.reshape(batch_size, -1).sum(-1).reshape(batch_size, 1, 1) - 40) / 10).exp() # For Intercap
        loss_obj_reproj = loss_obj_reproj.mean()

        return {
            'object_reproj_loss': loss_obj_reproj,
        }


class SMPLPostperioriLoss(nn.Module):

    def __init__(self, ):
        super(SMPLPostperioriLoss, self).__init__()


    def forward(self, hoi_dict):
        smpl_z = hoi_dict['smpl_z']
        smpl_postperiori_loss = (smpl_z ** 2).mean()

        return {
            'smpl_postperiori_loss': smpl_postperiori_loss
        }



class RelDistPostperioriLoss(nn.Module):

    def __init__(self, ):
        super(RelDistPostperioriLoss, self).__init__()


    def forward(self, hoi_dict):
        hoi_z = hoi_dict['hoi_z']
        hoi_postperiori_loss = (hoi_z ** 2).mean()

        return {
            'hoi_postperiori_loss': hoi_postperiori_loss
        }
