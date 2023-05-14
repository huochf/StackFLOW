import torch

from smplx import SMPLHLayer, SMPLXLayer

from stackflow.models.hoi_instance import HOIInstance
from stackflow.datasets.behave_metadata import load_object_mesh_templates as load_behave_object_mesh_templates
from stackflow.datasets.intercap_metadata import load_object_mesh_templates as load_intercap_object_mesh_templates
from stackflow.datasets.behave_metadata import OBJECT_IDX2NAME as BEHAVE_OBJECT_IDX2NAME
from stackflow.datasets.intercap_metadata import OBJECT_IDX2NAME as INTERCAP_OBJECT_IDX2NAME

from stackflow.utils.optimization import joint_optimize
from stackflow.utils.optimize_loss import SMPLPriorLoss, RelativeDistanceLoss


class HOIDecoder(object):

    def __init__(self, smpl_anchor_indices, object_anchor_indices, dataset='behave', fix_person=False, iterations=2, steps_per_iter=500, lr=0.01, device=torch.device('cuda')):
        self.smpl_anchor_indices = smpl_anchor_indices # (n)
        self.object_anchor_indices = object_anchor_indices # (#num_object, m, )
        self.dataset = dataset
        self.fix_person = fix_person
        self.iterations = iterations
        self.steps_per_iter = steps_per_iter
        self.lr = lr
        self.device = device
        if dataset == 'behave':
            self.smpl_type = 'smplh'
            self.smpl = SMPLHLayer(model_path='data/smplh', gender='male').to(device)
            self.object_templates = load_behave_object_mesh_templates()
            self.OBJECT_IDX2NAME = BEHAVE_OBJECT_IDX2NAME
        else:
            self.smpl_type = 'smplx'
            self.smpl = SMPLXLayer(model_path='data/smplx', gender='male').to(device)
            self.object_templates = load_intercap_object_mesh_templates()
            self.OBJECT_IDX2NAME = INTERCAP_OBJECT_IDX2NAME


    def forward(self, relative_distance, object_labels, smpl_init_params=None, ):

        if smpl_init_params is None:
            hoi_instance = HOIInstance(object_labels, smpl_type=self.smpl_type, dataset=self.dataset, fix_person=self.fix_person)
            hoi_instance.freeze_translation()
            hoi_instance.freeze_global_orient()
        else:
            smpl_trans_init = smpl_init_params['translation'] # (b, 10)
            smpl_beta_init = smpl_init_params['betas'] # (b, 10)
            smpl_global_orient_init = smpl_init_params['global_orient'] # (b, 3, 3)
            smpl_body_pose_init = smpl_init_params['body_pose'] # (b, 21, 3, 3)
            batch_size = smpl_trans_init.shape[0]

            smpl_out = self.smpl(betas=smpl_beta_init, 
                                 body_pose=smpl_body_pose_init, 
                                 global_orient=torch.eye(3, dtype=smpl_beta_init.dtype, device=smpl_beta_init.device).reshape(1, 3, 3).repeat(batch_size, 1, 1),
                                 transl=0 * smpl_trans_init)
            smpl_v = smpl_out.vertices

            n, m = self.smpl_anchor_indices.shape[0], self.object_anchor_indices.shape[1]
            person_anchors = smpl_v[:, self.smpl_anchor_indices].reshape(batch_size, n, 1, 3)
            ho_vectors = relative_distance.reshape(batch_size, n, m, 3)

            object_p = person_anchors + ho_vectors
            object_q = []
            for idx in range(batch_size):
                template_v = self.object_templates[self.OBJECT_IDX2NAME[object_labels[idx].item()]][0]
                object_q.append(template_v.to(object_p.device)[self.object_anchor_indices[object_labels[idx].item()]])
            object_q = torch.stack(object_q, dim=0)
            object_q = object_q.reshape(batch_size, 1, m, 3).repeat(1, n, 1, 1)
            P = object_p.reshape(batch_size, -1, 3)
            Q = object_q.reshape(batch_size, -1, 3)
            center_Q = Q.mean(1).reshape(batch_size, -1, 3)
            Q = Q - center_Q
            svd_mat = P.transpose(1, 2) @ Q
            svd_mat = svd_mat.double() # [b, 3, 3]
            u, _, v = torch.svd(svd_mat)
            d = torch.det(u @ v.transpose(1, 2)) # [b, ]
            d = torch.cat([
                torch.ones(batch_size, 2, device=u.device),
                d.unsqueeze(-1)], axis=-1) # [b, 3]
            d = torch.eye(3, device=u.device).unsqueeze(0) * d.view(-1, 1, 3)
            obj_rotmat = u @ d @ v.transpose(1, 2)
            obj_rotmat_pred = obj_rotmat.to(object_q.dtype) # (b * n, 3, 3)
            _Q = Q + center_Q
            obj_trans_pred = (P.transpose(1, 2) - obj_rotmat_pred @ _Q.transpose(1, 2)).mean(dim=2) # (n * b, 3)
            obj_rotmat_init = obj_rotmat_pred.reshape(batch_size, 3, 3)
            obj_trans_init = obj_trans_pred.reshape(batch_size, 3)

            hoi_instance = HOIInstance(object_labels=object_labels, 
                                       smpl_trans=smpl_trans_init, 
                                       smpl_global_orient=smpl_global_orient_init, 
                                       smpl_body_pose=smpl_body_pose_init,
                                       smpl_beta=smpl_beta_init,
                                       object_T_relative=obj_trans_init,
                                       object_R_relative=obj_rotmat_init,
                                       smpl_type=self.smpl_type,
                                       dataset=self.dataset,
                                       fix_person=self.fix_person)
        hoi_instance.to(self.device)
        loss_functions = [
            RelativeDistanceLoss(relative_distance, self.smpl_anchor_indices, self.object_anchor_indices[object_labels]).to(self.device), 
        ]
        loss_weights = {
            'rel_dist_loss': lambda cst, it: 10. ** 0 * cst / (1 + it),
        }
        joint_optimize(hoi_instance, loss_functions, loss_weights, iterations=self.iterations, steps_per_iter=self.steps_per_iter, lr=self.lr)
        return hoi_instance
