
import torch
import torch.nn as nn
from smplx import SMPLHLayer, SMPLXLayer

from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix


class HOIInstance(nn.Module):

    def __init__(self, smpl, J_regressor, object_v, smpl_betas=None, smpl_body_pose6d=None, obj_rel_trans=None, obj_rel_rotmat=None, hoi_trans=None, hoi_rot6d=None):
        super(HOIInstance, self).__init__()

        self.smpl = smpl
        self.register_buffer('object_v', object_v) # (bs, n, 3)
        self.register_buffer('J_regressor', torch.tensor(J_regressor, dtype=torch.float32).unsqueeze(0)) # (1, kps_num, n)
        batch_size = self.object_v.shape[0]

        if smpl_betas is not None:
            self.smpl_betas = nn.Parameter(smpl_betas.reshape(batch_size, 10))
        else:
            self.smpl_betas = nn.Parameter(torch.zeros(batch_size, 10, dtype=torch.float32))

        if smpl_body_pose6d is not None:
            self.smpl_body_pose6d = nn.Parameter(smpl_body_pose6d.reshape(batch_size, 21, 6))
        else:
            self.smpl_body_pose6d = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(batch_size, 21, 1, 1)))

        if obj_rel_trans is not None:
            self.obj_rel_trans = nn.Parameter(obj_rel_trans.reshape(batch_size, 3))
        else:
            self.obj_rel_trans = nn.Parameter(torch.zeros(batch_size, 3, dtype=torch.float32))

        if obj_rel_rotmat is not None:
            self.obj_rel_rot6d = nn.Parameter(matrix_to_rotation_6d(obj_rel_rotmat.reshape(batch_size, 3, 3)))
        else:
            self.obj_rel_rot6d = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).repeat(batch_size, 1, 1)))

        if hoi_trans is not None:
            self.hoi_trans = nn.Parameter(hoi_trans.reshape(batch_size, 3))
        else:
            self.hoi_trans = nn.Parameter(torch.zeros(batch_size, 3, dtype=torch.float32))

        if hoi_rot6d is not None:
            self.hoi_rot6d = nn.Parameter(hoi_rot6d.reshape(batch_size, 6))
        else:
            self.hoi_rot6d = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).repeat(batch_size, 1, 1)))


    def get_optimizer(self, fix_trans=False, fix_global_orient=False, fix_betas=True, lr=0.001):
        param_list = [self.smpl_body_pose6d, self.obj_rel_rot6d, self.obj_rel_trans]
        if not fix_trans:
            param_list.append(self.hoi_trans)
        if not fix_global_orient:
            param_list.append(self.hoi_rot6d)
        if not fix_betas:
            param_list.append(self.smpl_betas)

        optimizer = torch.optim.Adam(param_list, lr=lr, betas=(0.9, 0.999))
        return optimizer


    def forward(self, ):
        batch_size = self.smpl_betas.shape[0]
        smpl_body_rotmat = rotation_6d_to_matrix(self.smpl_body_pose6d)
        smpl_out = self.smpl(betas=self.smpl_betas, body_pose=smpl_body_rotmat)
        smpl_v = smpl_out.vertices
        smpl_J = smpl_out.joints # [:, :22]
        orig = smpl_J[:, 0:1]
        smpl_v = smpl_v - orig
        smpl_J = smpl_J - orig

        hoi_rotmat = rotation_6d_to_matrix(self.hoi_rot6d)
        smpl_v = torch.matmul(smpl_v, hoi_rotmat.permute(0, 2, 1)) + self.hoi_trans.reshape(batch_size, 1, 3)
        smpl_J = torch.matmul(smpl_J, hoi_rotmat.permute(0, 2, 1)) + self.hoi_trans.reshape(batch_size, 1, 3)

        openpose_kpts = torch.matmul(self.J_regressor, smpl_v)[:, :25]

        obj_rel_rotmat = rotation_6d_to_matrix(self.obj_rel_rot6d)
        obj_rotmat = torch.matmul(hoi_rotmat, obj_rel_rotmat)
        obj_trans = torch.matmul(hoi_rotmat, self.obj_rel_trans.reshape(batch_size, 3, 1)).squeeze(-1) + self.hoi_trans
        object_v = torch.matmul(self.object_v, obj_rotmat.permute(0, 2, 1)) + obj_trans.reshape(batch_size, 1, 3)

        results = {
            'smpl_betas': self.smpl_betas,
            'smpl_body_pose6d': self.smpl_body_pose6d,
            'smpl_body_rotmat': smpl_body_rotmat,
            'smpl_v': smpl_v,
            'smpl_J': smpl_J,
            'openpose_kpts': openpose_kpts,
            'obj_rel_trans': self.obj_rel_trans,
            'obj_rel_rotmat': obj_rel_rotmat,
            'obj_rel_rot6d': self.obj_rel_rot6d,
            'obj_rotmat': obj_rotmat,
            'obj_trans': obj_trans,
            'object_v': object_v,
            'hoi_rot6d': self.hoi_rot6d,
            'hoi_rotmat': hoi_rotmat,
            'hoi_trans': self.hoi_trans,
        }
        return results
