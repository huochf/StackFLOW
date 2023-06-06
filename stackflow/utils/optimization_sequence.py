import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from smplx import SMPLHLayer, SMPLXLayer
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix

from stackflow.datasets.utils import load_J_regressor
from stackflow.utils.camera import perspective_projection


class HOIInstance(nn.Module):

    def __init__(self, smpl, J_regressor, batch_size, smpl_betas=None, smpl_body_pose6d=None, obj_rel_trans=None, obj_rel_rotmat=None, hoi_trans=None, hoi_rot6d=None):
        super(HOIInstance, self).__init__()

        self.smpl = smpl
        self.register_buffer('J_regressor', torch.tensor(J_regressor, dtype=torch.float32).unsqueeze(0)) # (1, kps_num, n)

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


    def get_optimizer(self, fix_trans=False, fix_global_orient=False, lr=0.001, other_params=[]):
        param_list = [self.smpl_betas, self.smpl_body_pose6d, self.obj_rel_rot6d, self.obj_rel_trans, ] + other_params
        if not fix_trans:
            param_list.append(self.hoi_trans)
        if not fix_global_orient:
            param_list.append(self.hoi_rot6d)

        optimizer = torch.optim.Adam(param_list, lr=lr, betas=(0.9, 0.999))
        return optimizer


    def forward(self, begin_idx, end_idx, object_v=None):
        smpl_body_rotmat = rotation_6d_to_matrix(self.smpl_body_pose6d[begin_idx:end_idx])
        smpl_out = self.smpl(betas=self.smpl_betas[begin_idx:end_idx], body_pose=smpl_body_rotmat)
        smpl_v = smpl_out.vertices
        smpl_J = smpl_out.joints # [:, :22]
        orig = smpl_J[:, 0:1]
        smpl_v = smpl_v - orig
        smpl_J = smpl_J - orig

        hoi_rotmat = rotation_6d_to_matrix(self.hoi_rot6d[begin_idx:end_idx])
        smpl_v = torch.matmul(smpl_v, hoi_rotmat.permute(0, 2, 1)) + self.hoi_trans[begin_idx:end_idx].reshape(-1, 1, 3)
        smpl_J = torch.matmul(smpl_J, hoi_rotmat.permute(0, 2, 1)) + self.hoi_trans[begin_idx:end_idx].reshape(-1, 1, 3)

        openpose_kpts = torch.matmul(self.J_regressor, smpl_v)[:, :25]

        obj_rel_rotmat = rotation_6d_to_matrix(self.obj_rel_rot6d[begin_idx:end_idx])
        obj_rotmat = torch.matmul(hoi_rotmat, obj_rel_rotmat)
        obj_trans = torch.matmul(hoi_rotmat, self.obj_rel_trans[begin_idx:end_idx].reshape(-1, 3, 1)).squeeze(-1) + self.hoi_trans[begin_idx:end_idx]
        if object_v is not None:
            object_v = torch.matmul(object_v, obj_rotmat.permute(0, 2, 1)) + obj_trans.reshape(-1, 1, 3)

        results = {
            'smpl_betas': self.smpl_betas[begin_idx:end_idx],
            'smpl_body_pose6d': self.smpl_body_pose6d[begin_idx:end_idx],
            'smpl_body_rotmat': smpl_body_rotmat,
            'smpl_v': smpl_v,
            'smpl_J': smpl_J,
            'openpose_kpts': openpose_kpts,
            'obj_rel_trans': self.obj_rel_trans[begin_idx:end_idx],
            'obj_rel_rotmat': obj_rel_rotmat,
            'obj_rel_rot6d': self.obj_rel_rot6d[begin_idx:end_idx],
            'obj_rotmat': obj_rotmat,
            'obj_trans': obj_trans,
            'object_v': object_v,
            'hoi_rot6d': self.hoi_rot6d[begin_idx:end_idx],
            'hoi_rotmat': hoi_rotmat,
            'hoi_trans': self.hoi_trans[begin_idx:end_idx],
        }
        return results


def object_reprojection_loss(hoi_dict, obj_x3d, obj_x2d, obj_w2d, focal_length, optical_center):
    obj_rotmat = hoi_dict['obj_rotmat']
    obj_trans = hoi_dict['obj_trans']
    b = obj_trans.shape[0]
    reproj_points = perspective_projection(points=obj_x3d, trans=obj_trans, rotmat=obj_rotmat, 
        focal_length=focal_length, optical_center=optical_center)

    loss_obj_reproj = F.l1_loss(obj_x2d, reproj_points, reduction='none') * obj_w2d
    loss_obj_reproj = loss_obj_reproj.reshape(b, -1).mean(-1)

    return {
        'object_reproj_loss': loss_obj_reproj,
    }


def person_keypoints_loss(hoi_dict, person_keypoints, focal_length, optical_center):
    openpose_kpts = hoi_dict['openpose_kpts']
    batch_size = openpose_kpts.shape[0]
    kpts_reproj = perspective_projection(points=openpose_kpts, focal_length=focal_length, optical_center=optical_center)

    loss_kpts_reproj = F.l1_loss(kpts_reproj, person_keypoints[:, :, :2], reduction='none') * person_keypoints[:, :, 2:]
    loss = loss_kpts_reproj.reshape(batch_size, -1).mean(-1)


    return {
        'person_reproj_loss': loss,
    }


def posterior_loss(hoi_dict, stackflow, hooffset, human_features, hoi_features, object_labels):
    smpl_orient = hoi_dict['hoi_rot6d']
    smpl_body_pose6d = hoi_dict['smpl_body_pose6d']
    b = smpl_orient.shape[0]
    smpl_pose6d = torch.cat([smpl_orient.unsqueeze(1), smpl_body_pose6d], dim=1).reshape(b, -1)

    smpl_betas = hoi_dict['smpl_betas']
    obj_rel_rotmat = hoi_dict['obj_rel_rotmat']
    obj_rel_trans = hoi_dict['obj_rel_trans']
    smpl_body_rotmat = hoi_dict['smpl_body_rotmat']
    gamma, _ = hooffset.encode(smpl_betas, smpl_body_rotmat, obj_rel_rotmat, obj_rel_trans, object_labels)

    theta_log_prob, theta_z, gamma_log_prob, gamma_z = stackflow.log_prob(smpl_pose6d, gamma, human_features, hoi_features, object_labels)

    smpl_pose_posterior_loss = (theta_z ** 2).reshape(b, -1).mean(-1)
    offset_posterior_loss = (gamma_z ** 2).reshape(b, -1).mean(-1)
    return {
        'smpl_pose_posterior_loss': smpl_pose_posterior_loss,
        'offset_posterior_loss': offset_posterior_loss,
    }


def any_smooth_loss(data):
    # data: [n, c]
    # return ((data[:-2] + data[2:] - 2 * data[1:-1]) ** 2).mean(-1) + 0.01 * ((data[1:] - data[:-1]) ** 2).mean(-1)
    loss = 100 * ((data[1:] - data[:-1]) ** 2).mean().reshape(1, 1) # strong smooth, for better visualization
    for idx, windows in enumerate([1, 2, 3]):
        loss += 0.5 ** idx * ((data[:-2 * windows] + data[2 * windows:] - 2 * data[windows:-windows]) ** 2).mean().reshape(1, 1)
    return loss

def sequence_smooth_loss(hoi_dict, ):
    smpl_betas = hoi_dict['smpl_betas']
    obj_rel_rotmat = hoi_dict['obj_rel_rotmat']
    obj_rel_trans = hoi_dict['obj_rel_trans']
    smpl_body_rotmat = hoi_dict['smpl_body_rotmat']
    hoi_rot6d = hoi_dict['hoi_rot6d']
    hoi_trans = hoi_dict['hoi_trans']

    b = smpl_betas.shape[0]
    smpl_betas_smooth_loss = any_smooth_loss(smpl_betas.reshape(b, -1))
    smpl_body_rotmat_smooth_loss = any_smooth_loss(smpl_body_rotmat.reshape(b, -1))
    obj_rel_rotmat_smooth_loss = any_smooth_loss(obj_rel_rotmat.reshape(b, -1))
    obj_rel_trans_smooth_loss = any_smooth_loss(obj_rel_trans.reshape(b, -1))
    hoi_rot6d_smooth_loss = any_smooth_loss(hoi_rot6d.reshape(b, -1))
    hoi_trans_smooth_loss = any_smooth_loss(hoi_trans.reshape(b, -1))

    smpl_v = hoi_dict['smpl_v']
    object_v = hoi_dict['object_v']
    smpl_v_smooth_loss = any_smooth_loss(smpl_v.reshape(b, -1))
    object_v_smooth_loss = any_smooth_loss(object_v.reshape(b, -1))

    return {
        'smpl_v_smooth_loss': smpl_v_smooth_loss,
        'object_v_smooth_loss': object_v_smooth_loss,
        'smpl_betas_smooth_loss': smpl_betas_smooth_loss,
        'smpl_body_rotmat_smooth_loss': smpl_body_rotmat_smooth_loss,
        'obj_rel_rotmat_smooth_loss': obj_rel_rotmat_smooth_loss,
        'obj_rel_trans_smooth_loss': obj_rel_trans_smooth_loss,
        'hoi_rot6d_smooth_loss': hoi_rot6d_smooth_loss,
        'hoi_trans_smooth_loss': hoi_trans_smooth_loss,
    }


def get_loss_weights():
    return {
        'person_reproj_loss': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
        'object_reproj_loss':  lambda cst, it: 10. ** 1 * cst / (1 + 10 * it),
        'smpl_pose_posterior_loss': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
        'offset_posterior_loss':  lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),

        'smpl_betas_smooth_loss': lambda cst, it: 10. ** 2 * cst / (1 + 10 * it),
        'smpl_body_rotmat_smooth_loss': lambda cst, it: 10. ** 2 * cst / (1 + 10 * it),
        'obj_rel_rotmat_smooth_loss': lambda cst, it: 10. ** 2 * cst / (1 + 10 * it),
        'obj_rel_trans_smooth_loss': lambda cst, it: 10. ** 2 * cst / (1 + 10 * it),
        'hoi_rot6d_smooth_loss': lambda cst, it: 10. ** 2 * cst / (1 + 10 * it),
        'hoi_trans_smooth_loss': lambda cst, it: 10. ** 2 * cst / (1 + 10 * it),
        'smpl_v_smooth_loss': lambda cst, it: 10. ** 2 * cst / (1 + 10 * it),
        'object_v_smooth_loss': lambda cst, it: 10. ** 2 * cst / (1 + 10 * it),
    }


def post_optimization_sequence(cfg, dataset_metadata, model, all_batch, all_predictions):
    device = torch.device('cuda')

    if cfg.dataset.name == 'BEHAVE' or cfg.dataset.name == 'BEHAVE-Extended':
        smpl = SMPLHLayer(model_path=cfg.model.smplh_dir, gender='male').to(device)
        J_regressor = load_J_regressor(cfg.model.smplh_regressor)
    elif cfg.dataset.name == 'InterCap':
        smpl = SMPLXLayer(model_path=cfg.model.smplx_dir, gender='neutral').to(device)
        J_regressor = load_J_regressor(cfg.model.smplx_regressor)
    else:
        assert False

    num_object = len(dataset_metadata.OBJECT_IDX2NAME)
    object_template_v = np.zeros((num_object, dataset_metadata.object_max_vertices_num, 3))
    for idx, object_idx in enumerate(sorted(dataset_metadata.OBJECT_IDX2NAME.keys())):
        object_name = dataset_metadata.OBJECT_IDX2NAME[object_idx]
        vertices = dataset_metadata.obj_mesh_templates[object_name][0]
        object_template_v[idx, :len(vertices)] = vertices
    object_template_v = torch.tensor(object_template_v, dtype=torch.float32).to(device)

    num_sequences = all_predictions['pred_betas'].shape[0]
    hoi_instance = HOIInstance(smpl=smpl,
                               J_regressor=J_regressor,
                               batch_size=num_sequences,
                               smpl_betas=all_predictions['pred_betas'].detach(), 
                               smpl_body_pose6d=all_predictions['pred_pose6d'].detach()[:, 1:], 
                               obj_rel_trans=all_predictions['pred_obj_rel_T'].detach(), 
                               obj_rel_rotmat=all_predictions['pred_obj_rel_R'].detach(), 
                               hoi_trans=all_predictions['hoi_trans'].detach(), 
                               hoi_rot6d=all_predictions['pred_pose6d'].detach()[:, 0]).to(device)

    loss_weights = get_loss_weights()
    iterations = cfg.eval.optim_iters
    if cfg.dataset.name == 'BEHAVE-Extended':
        steps_per_iter = 100 # cfg.eval.optim_steps
        lr = 2e-3
    elif cfg.dataset.name == 'InterCap':
        steps_per_iter = 200
        lr = 5e-3
    batch_size = 128

    optimizer = hoi_instance.get_optimizer(fix_trans=False, fix_global_orient=False, lr=lr) # cfg.eval.optim_lr)
    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        for i in loop:
            all_losses = {}

            def append(dict1, dict2):
                for k in dict2:
                    if k not in dict1:
                        dict1[k] = dict2[k]
                    else:
                        dict1[k] = torch.cat([dict1[k], dict2[k]], dim=0)
                return dict1

            for batch_idx in range(0, num_sequences, batch_size // 2):
                object_labels = all_batch['object_labels'][batch_idx:batch_idx + batch_size].to(device)
                object_v = object_template_v[object_labels]
                person_keypoints = all_batch['person_kps'][batch_idx:batch_idx + batch_size].to(device)
                obj_x3d = all_batch['obj_x3d'][batch_idx:batch_idx + batch_size].to(device)
                obj_x2d = all_batch['obj_x2d'][batch_idx:batch_idx + batch_size].to(device)
                obj_w2d = all_batch['obj_w2d'][batch_idx:batch_idx + batch_size].to(device)
                focal_length = all_batch['focal_length'][batch_idx:batch_idx + batch_size].to(device)
                optical_center = all_batch['optical_center'][batch_idx:batch_idx + batch_size].to(device)

                human_features = all_predictions['human_features'][batch_idx:batch_idx + batch_size].to(device)
                hoi_features = all_predictions['hoi_features'][batch_idx:batch_idx + batch_size].to(device)

                optimizer.zero_grad()
                hoi_dict = hoi_instance.forward(batch_idx, batch_idx + batch_size, object_v)
                losses = {}

                losses.update(object_reprojection_loss(hoi_dict, obj_x3d, obj_x2d, obj_w2d, focal_length, optical_center))
                losses.update(person_keypoints_loss(hoi_dict, person_keypoints, focal_length, optical_center))
                losses.update(posterior_loss(hoi_dict, model.stackflow, model.flow_loss.hooffset, human_features, hoi_features, object_labels))
                losses.update(sequence_smooth_loss(hoi_dict))

                loss_list = [loss_weights[k](v.sum(), it) for k, v in losses.items()]
                total_loss = torch.stack(loss_list).sum()

                total_loss.backward()
                optimizer.step()

                all_losses = append(all_losses, losses)

            l_str = 'Optim. Step {}: Iter: {}'.format(it, i)
            for k, v in all_losses.items():
                l_str += ', {}: {:0.4f}'.format(k, v.mean().detach().item())
                loop.set_description(l_str)

    hoi_dict = hoi_instance.forward(0, num_sequences)
    all_predictions['pred_betas'] = hoi_dict['smpl_betas']
    all_predictions['pred_smpl_body_pose'] = hoi_dict['smpl_body_rotmat']
    all_predictions['pred_obj_rel_T'] = hoi_dict['obj_rel_trans']
    all_predictions['pred_obj_rel_R'] = hoi_dict['obj_rel_rotmat']
    all_predictions['hoi_trans'] = hoi_dict['hoi_trans']
    all_predictions['hoi_rotmat'] = hoi_dict['hoi_rotmat']

    return all_predictions
