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

    def __init__(self, object_labels=None, smpl_trans=None, smpl_global_orient=None, smpl_body_pose=None, smpl_beta=None,
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

        if smpl_type == 'smplh':
            self.smpl = SMPLHLayer(model_path='data/smplh', gender='male')
        else:
            self.smpl = SMPLXLayer(model_path='data/smplx', gender='male')

        if smpl_beta is not None:
            self.smpl_beta = nn.Parameter(smpl_beta.reshape(batch_size, 10))
        else:
            self.smpl_beta = nn.Parameter(torch.zeros(batch_size, 10))

        if smpl_body_pose is not None:
            self.smpl_body_pose = nn.Parameter(matrix_to_axis_angle(smpl_body_pose.reshape(batch_size, -1, 3, 3)[:, :21]).reshape(batch_size, 63))
        else:
            pose_init = torch.zeros(batch_size, 21, 3, dtype=torch.float32)
            self.smpl_body_pose = nn.Parameter(pose_init.reshape(batch_size, 63))

        if smpl_global_orient is not None:
            self.smpl_global_orient = nn.Parameter(matrix_to_axis_angle(smpl_global_orient.reshape(batch_size, 3, 3)))
        else:
            global_orient = torch.zeros(batch_size, 3, dtype=torch.float32)
            self.smpl_global_orient = nn.Parameter(global_orient)

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


    def freeze_translation(self, ):
        self.smpl_trans.requires_grad = False


    def freeze_global_orient(self, ):
        self.smpl_global_orient.requires_grad = False


    def get_optimizer(self, fix_smpl_trans=False, fix_smpl_global_orient=False, lr=0.001):
        param_list = [self.smpl_beta, self.smpl_body_pose, self.object_T_relative, self.object_R6d_relative]
        if not fix_smpl_trans:
            param_list.append(self.smpl_trans)
        if not fix_smpl_global_orient:
            param_list.append(self.smpl_global_orient)

        if self.fix_person:
            param_list = [self.object_T_relative, self.object_R6d_relative]

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(param_list, lr=lr, betas=(0.9, 0.999))
        return self.optimizer


    def forward(self, ):
        batch_size = self.smpl_beta.shape[0]
        smpl_out = self.smpl(betas=self.smpl_beta, body_pose=axis_angle_to_matrix(self.smpl_body_pose[:, :63].reshape(batch_size, 21, 3)), 
            global_orient=axis_angle_to_matrix(self.smpl_global_orient).reshape(batch_size, 3, 3), transl=self.smpl_trans)
        smpl_out_orig = self.smpl(betas=self.smpl_beta, body_pose=axis_angle_to_matrix(self.smpl_body_pose[:, :63].reshape(batch_size, 21, 3)), 
            global_orient=torch.eye(3, dtype=self.smpl_global_orient.dtype, device=self.smpl_global_orient.device).reshape(1, 3, 3).repeat(batch_size, 1, 1), 
            transl=0.*self.smpl_trans)

        root_rotmat = axis_angle_to_matrix(self.smpl_global_orient)
        J = smpl_out_orig.joints.detach()
        root_T = self.smpl_trans - (torch.einsum('bst,bvt->bvs', root_rotmat, J[:, 0:1]) - J[:, 0:1]).squeeze(1)
        root_T = root_T.reshape(batch_size, 3, )

        object_v = self.object_v
        object_v_orig = compute_transformation_persp(object_v, self.object_T_relative.reshape(batch_size, 1, 3), self.object_R6d_relative)

        object_T = self.object_T_relative.unsqueeze(1) @ axis_angle_to_matrix(self.smpl_global_orient).permute(0, 2, 1) + root_T.unsqueeze(1)
        object_R6d = matrix_to_rotation_6d(axis_angle_to_matrix(self.smpl_global_orient) @ rotation_6d_to_matrix(self.object_R6d_relative))
        object_v = compute_transformation_persp(object_v, object_T.reshape(batch_size, 1, 3), object_R6d)

        results = {
            'smpl_v': smpl_out.vertices,
            'object_v': object_v,
            'smpl_v_orig': smpl_out_orig.vertices,
            'object_v_orig': object_v_orig,
            'smpl_betas': self.smpl_beta,
            'smpl_body_pose': self.smpl_body_pose.reshape(batch_size, 63),
            'smpl_trans': self.smpl_trans,
            'smpl_global_orient': self.smpl_global_orient,
            'object_global_rotmat': rotation_6d_to_matrix(object_R6d),
            'object_global_trans': object_T,
            'object_relative_R': rotation_6d_to_matrix(self.object_R6d_relative),
            'object_relative_T': self.object_T_relative,
            'object_v_template': self.object_v,
            'smpl_J': smpl_out.joints,
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
