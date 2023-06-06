import torch
import torch.nn as nn
from smplx import SMPLHLayer, SMPLXLayer

from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix


class HOIInstance(nn.Module):

    def __init__(self, smpl, J_regressor, object_v, object_labels, stackflow, hooffset, human_features, hoi_features, smpl_betas=None, hoi_trans=None,):
        super(HOIInstance, self).__init__()

        self.smpl = smpl
        self.register_buffer('object_v', object_v) # (bs, n, 3)
        self.register_buffer('J_regressor', torch.tensor(J_regressor, dtype=torch.float32).unsqueeze(0)) # (1, kps_num, n)
        batch_size = self.object_v.shape[0]

        self.register_buffer('object_labels', object_labels)
        self.register_buffer('human_features', human_features)
        self.register_buffer('hoi_features', hoi_features)

        self.stackflow = stackflow
        self.hooffset = hooffset

        if smpl_betas is not None:
            self.smpl_betas = nn.Parameter(smpl_betas.reshape(batch_size, 10))
        else:
            self.smpl_betas = nn.Parameter(torch.zeros(batch_size, 10, dtype=torch.float32))

        if hoi_trans is not None:
            self.hoi_trans = nn.Parameter(hoi_trans.reshape(batch_size, 3))
        else:
            self.hoi_trans = nn.Parameter(torch.zeros(batch_size, 3, dtype=torch.float32))

        z_theta_dim = (21 + 1) * 6
        z_gamma_dim = 32 
        self.theta_z = nn.Parameter(torch.zeros(batch_size, z_theta_dim, dtype=torch.float32))
        self.gamma_z = nn.Parameter(torch.zeros(batch_size, z_gamma_dim, dtype=torch.float32))


    def get_optimizer(self, lr=0.001):
        param_list = [self.smpl_betas, self.hoi_trans, self.theta_z, self.gamma_z]
        optimizer = torch.optim.Adam(param_list, lr=lr, betas=(0.9, 0.999))
        return optimizer


    def forward(self, ):
        batch_size = self.smpl_betas.shape[0]

        theta, _, _, gamma, _, _ = self.stackflow(self.human_features, self.hoi_features, object_labels=self.object_labels, theta_z=self.theta_z, gamma_z=self.gamma_z)
        smpl_pose6d = theta.reshape(batch_size, -1, 6)
        smpl_rotmat = rotation_6d_to_matrix(smpl_pose6d)

        smpl_out = self.smpl(betas=self.smpl_betas, body_pose=smpl_rotmat[:, 1:])
        smpl_v = smpl_out.vertices
        smpl_J = smpl_out.joints # [:, :22]
        orig = smpl_J[:, 0:1]
        smpl_v = smpl_v - orig
        smpl_J = smpl_J - orig

        hoi_rotmat = smpl_rotmat[:, 0]
        smpl_v = torch.matmul(smpl_v, hoi_rotmat.permute(0, 2, 1)) + self.hoi_trans.reshape(batch_size, 1, 3)
        smpl_J = torch.matmul(smpl_J, hoi_rotmat.permute(0, 2, 1)) + self.hoi_trans.reshape(batch_size, 1, 3)

        openpose_kpts = torch.matmul(self.J_regressor, smpl_v)[:, :25]

        offsets = self.hooffset.decode(gamma, self.object_labels)
        obj_rel_rotmat, obj_rel_trans = self.hooffset.decode_object_RT(offsets, self.smpl_betas, smpl_rotmat[:, 1:], self.object_labels)

        obj_rotmat = torch.matmul(hoi_rotmat, obj_rel_rotmat)
        obj_trans = torch.matmul(hoi_rotmat, obj_rel_trans.reshape(batch_size, 3, 1)).squeeze(-1) + self.hoi_trans
        object_v = torch.matmul(self.object_v, obj_rotmat.permute(0, 2, 1)) + obj_trans.reshape(batch_size, 1, 3)

        results = {
            'smpl_betas': self.smpl_betas,
            'smpl_body_pose6d': smpl_pose6d[:, 1:],
            'smpl_body_rotmat': smpl_rotmat[:, 1:],
            'smpl_v': smpl_v,
            'smpl_J': smpl_J,
            'openpose_kpts': openpose_kpts,
            'obj_rel_trans': obj_rel_trans,
            'obj_rel_rotmat': obj_rel_rotmat,
            'obj_rel_rot6d': matrix_to_rotation_6d(obj_rel_rotmat),
            'obj_rotmat': obj_rotmat,
            'obj_trans': obj_trans,
            'object_v': object_v,
            'hoi_rot6d': smpl_pose6d[:, 0],
            'hoi_rotmat': smpl_rotmat[:, 0],
            'hoi_trans': self.hoi_trans,
            'theta_z': self.theta_z,
            'gamma_z': self.gamma_z,
        }
        return results
