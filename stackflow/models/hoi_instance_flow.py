import torch
import torch.nn as nn
from smplx import SMPLHLayer, SMPLXLayer

from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix

from stackflow.datasets.behave_metadata import MAX_OBJECT_VERT_NUM as BEHAVE_MAX_OBJECT_VERT_NUM
from stackflow.datasets.behave_metadata import OBJECT_IDX2NAME as BEHAVE_OBJECT_IDX2NAME
from stackflow.datasets.behave_metadata import load_object_mesh_templates as load_behave_object_mesh_templates
from stackflow.datasets.intercap_metadata import MAX_OBJECT_VERT_NUM as INTERCAP_MAX_OBJECT_VERT_NUM
from stackflow.datasets.intercap_metadata import OBJECT_IDX2NAME as INTERCAP_OBJECT_IDX2NAME
from stackflow.datasets.intercap_metadata import load_object_mesh_templates as load_intercap_object_mesh_templates


class HOIInstance(nn.Module):

    def __init__(self, hoi_flow, object_labels, condition_features, smpl_trans=None,
        object_T_relative=None, object_R_relative=None, smpl_type='smplh', dataset='behave', fix_person=False, device=torch.device('cuda')):
        super(HOIInstance, self).__init__()
        if dataset == 'behave':
            load_object_mesh_templates = load_behave_object_mesh_templates
            MAX_OBJECT_VERT_NUM = BEHAVE_MAX_OBJECT_VERT_NUM
            OBJECT_IDX2NAME = BEHAVE_OBJECT_IDX2NAME
        else:
            load_object_mesh_templates = load_intercap_object_mesh_templates
            MAX_OBJECT_VERT_NUM = INTERCAP_MAX_OBJECT_VERT_NUM
            OBJECT_IDX2NAME = INTERCAP_OBJECT_IDX2NAME
        self.object_mesh_templates = load_object_mesh_templates()
        object_v = []
        batch_size = object_labels.shape[0]
        max_points = MAX_OBJECT_VERT_NUM
        self.object_v = torch.zeros(batch_size, max_points, 3, dtype=torch.float32, device=device)
        for i in range(batch_size):
            v, _ = self.object_mesh_templates[OBJECT_IDX2NAME[int(object_labels[i])]]
            self.object_v[i, :len(v)] = v

        self.hoi_flow = hoi_flow
        self.object_labels = object_labels
        self.condition_features = condition_features

        if smpl_type == 'smplh':
            self.smpl = SMPLHLayer(model_path='data/smplh', gender='male')
        else:
            self.smpl = SMPLXLayer(model_path='data/smplx', gender='male')

        self.smpl_z = nn.Parameter(torch.zeros(batch_size, hoi_flow.smpl_pose))
        self.hoi_z = nn.Parameter(torch.zeros(batch_size, hoi_flow.hoi_dim))

        if smpl_trans is not None:
            self.smpl_trans = nn.Parameter(smpl_trans.reshape(batch_size, 3))
        else:
            self.smpl_trans = nn.Parameter(torch.zeros(batch_size, 3))
            
        if object_T_relative is not None:
            self.object_T_relative = nn.Parameter(object_T_relative.reshape(batch_size, 3))
        else:
            self.object_T_relative = nn.Parameter(torch.zeros(batch_size, 3))

        if object_R_relative is not None:
            self.object_R6d_relative = nn.Parameter(matrix_to_rotation_6d(object_R_relative.reshape(batch_size, 3, 3)))
        else:
            self.object_R6d_relative = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)))

        self.fix_person = fix_person
        self.optimizer = None


    def get_optimizer(self, fix_smpl_trans=False, fix_smpl_global_orient=False, lr=0.001):
        param_list = [self.smpl_z, self.hoi_z, self.object_T_relative, self.object_R6d_relative]
        if not fix_smpl_trans:
            param_list.append(self.smpl_trans)

        if self.fix_person:
            param_list = [self.hoi_z, self.object_T_relative, self.object_R6d_relative]

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(param_list, lr=lr, betas=(0.9, 0.999))
        return self.optimizer


    def forward(self, ):
        batch_size = self.smpl_z.shape[0]
        z = (self.smpl_z.unsqueeze(1), self.hoi_z.unsqueeze(1))
        smpl_samples, hoi_samples, _ = self.hoi_flow(features=self.condition_features, object_labels=self.object_labels, z=z)
        pose_rotmat = smpl_samples['pose_rotmat'].reshape(batch_size, 22, 3, 3)
        betas = smpl_samples['betas'].reshape(batch_size, 10)
        hoi_lattent_codes = hoi_samples['hoi_lattent_codes'].reshape(batch_size, -1)

        smpl_out = self.smpl(betas=betas, body_pose=pose_rotmat[:, 1:22], global_orient=pose_rotmat[:, 0:1], transl=self.smpl_trans)
        smpl_out_orig = self.smpl(betas=betas, body_pose=pose_rotmat[:, 1:22], 
            global_orient=torch.eye(3, dtype=pose_rotmat.dtype, device=pose_rotmat.device).reshape(1, 3, 3).repeat(batch_size, 1, 1), transl=0*self.smpl_trans)

        root_rotmat = pose_rotmat[:, 0]
        J = smpl_out_orig.joints.detach()
        root_T = self.smpl_trans - (torch.einsum('bst,bvt->bvs', root_rotmat, J[:, 0:1]) - J[:, 0:1]).squeeze(1)
        root_T = root_T.reshape(batch_size, 3, )

        object_v = self.object_v
        object_v_orig = compute_transformation_persp(object_v, self.object_T_relative.reshape(batch_size, 1, 3), self.object_R6d_relative)

        object_T = self.object_T_relative.unsqueeze(1) @ pose_rotmat[:, 0].permute(0, 2, 1) + root_T.unsqueeze(1)
        object_R6d = matrix_to_rotation_6d(pose_rotmat[:, 0] @ rotation_6d_to_matrix(self.object_R6d_relative))
        object_v = compute_transformation_persp(object_v, object_T.reshape(batch_size, 1, 3), object_R6d)

        results = {
            "smpl_v": smpl_out.vertices,
            "object_v": object_v,
            'smpl_v_orig': smpl_out_orig.vertices,
            'object_v_orig': object_v_orig,
            'smpl_betas': betas,
            'smpl_body_pose': matrix_to_axis_angle(pose_rotmat[:, 1:22]).reshape(batch_size, 63),
            'smpl_trans': self.smpl_trans,
            'smpl_global_orient': matrix_to_axis_angle(pose_rotmat[:, 0]),
            'object_global_rotmat': rotation_6d_to_matrix(object_R6d),
            'object_global_trans': object_T,
            'object_relative_R': rotation_6d_to_matrix(self.object_R6d_relative),
            'object_relative_T': self.object_T_relative,
            'object_v_template': self.object_v,
            'smpl_J': smpl_out.joints,
            'smpl_z': self.smpl_z,
            'hoi_z': self.hoi_z,
            'hoi_lattent_codes': hoi_lattent_codes,
        }
        return results


def compute_transformation_persp(meshes, translations, rotations=None):
    """
    Computes the 3D transformation.

    Args:
        meshes (V x 3 or B x V x 3): Vertices.
        translations (B x 1 x 3).
        rotations (B x 3 x 3).
    Returns:
        vertices (B x V x 3)
    """
    B = translations.shape[0]
    device = meshes.device
    if meshes.ndimension() == 2:
        meshes = meshes.repeat(B, 1, 1)
    if rotations is None:
        rotations = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotations = rotations.to(device)
    rotations = rotation_6d_to_matrix(rotations).transpose(2, 1)
    verts_rot = torch.matmul(meshes.detach().clone(), rotations)
    verts_final = verts_rot + translations

    return verts_final
