import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.transforms import matrix_to_rotation_6d

from stackflow.utils.camera import perspective_projection


class ObjectReprojLoss(nn.Module):

    def __init__(self, model_points, image_points, pts_confidence, focal_length, optical_center, rescale=True):
        super(ObjectReprojLoss, self).__init__()
        self.register_buffer('model_points', model_points)
        self.register_buffer('image_points', image_points)
        self.register_buffer('pts_confidence', pts_confidence)
        self.register_buffer('focal_length', focal_length)
        self.register_buffer('optical_center', optical_center)
        b = model_points.shape[0]
        self.register_buffer('weights', torch.ones(b, dtype=torch.float32))

        self.rescale = rescale


    def set_weights(self, weights):
        self.weights = weights


    def forward(self, hoi_dict):
        obj_rotmat = hoi_dict['obj_rotmat']
        obj_trans = hoi_dict['obj_trans']
        b = obj_rotmat.shape[0]
        reproj_points = perspective_projection(points=self.model_points, trans=obj_trans, rotmat=obj_rotmat, 
            focal_length=self.focal_length, optical_center=self.optical_center)

        loss_obj_reproj = F.l1_loss(self.image_points, reproj_points, reduction='none') * self.pts_confidence
        loss_obj_reproj = loss_obj_reproj.reshape(b, -1).mean(-1)
        loss_obj_reproj = loss_obj_reproj * self.weights

        return {
            'object_reproj_loss': loss_obj_reproj,
        }


class PersonKeypointLoss(nn.Module):

    def __init__(self, keypoints, confidences, focal_length, optical_center, loss_type='l1'):
        super(PersonKeypointLoss, self).__init__()
        self.register_buffer('keypoints', keypoints)
        self.register_buffer('confidences', confidences)
        self.register_buffer('focal_length', focal_length)
        self.register_buffer('optical_center', optical_center)
        self.loss_type = loss_type


    def forward(self, hoi_dict):
        openpose_kpts = hoi_dict['openpose_kpts']
        batch_size = openpose_kpts.shape[0]
        kpts_reproj = perspective_projection(points=openpose_kpts, focal_length=self.focal_length, optical_center=self.optical_center)

        if self.loss_type == 'l1':
            loss_kpts_reproj = F.l1_loss(kpts_reproj, self.keypoints, reduction='none') * self.confidences
            loss = loss_kpts_reproj.reshape(batch_size, -1).mean(-1)


        return {
            'person_reproj_loss': loss,
        }


class PosteriorLoss(nn.Module):

    def __init__(self, stackflow, hooffset, human_features, hoi_features, object_labels):
        super(PosteriorLoss, self).__init__()
        self.stackflow = stackflow
        self.hooffset = hooffset
        self.register_buffer('human_features', human_features)
        self.register_buffer('hoi_features', hoi_features)
        self.register_buffer('object_labels', object_labels)


    def forward(self, hoi_dict):
        smpl_orient = hoi_dict['hoi_rot6d']
        smpl_body_pose6d = hoi_dict['smpl_body_pose6d']
        b = smpl_orient.shape[0]
        smpl_pose6d = torch.cat([smpl_orient.unsqueeze(1), smpl_body_pose6d], dim=1).reshape(b, -1)

        smpl_betas = hoi_dict['smpl_betas']
        obj_rel_rotmat = hoi_dict['obj_rel_rotmat']
        obj_rel_trans = hoi_dict['obj_rel_trans']
        smpl_body_rotmat = hoi_dict['smpl_body_rotmat']
        gamma, _ = self.hooffset.encode(smpl_betas, smpl_body_rotmat, obj_rel_rotmat, obj_rel_trans, self.object_labels)

        theta_log_prob, theta_z, gamma_log_prob, gamma_z = self.stackflow.log_prob(smpl_pose6d, gamma, self.human_features, self.hoi_features, self.object_labels)

        smpl_pose_posterior_loss = (theta_z ** 2).reshape(b, -1).mean(-1)
        offset_posterior_loss = (gamma_z ** 2).reshape(b, -1).mean(-1)
        return {
            'smpl_pose_posterior_loss': smpl_pose_posterior_loss,
            'offset_posterior_loss': offset_posterior_loss,
        }
