import torch
import torch.nn as nn

from smplx import SMPLHLayer, SMPLXLayer

from stackflow.datasets.behave_metadata import BEHAVEMetaData
from stackflow.datasets.intercap_metadata import InterCapMetaData
from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from stackflow.datasets.utils import load_pickle


class HOOffset(nn.Module):

    def __init__(self, cfg):
        super(HOOffset, self).__init__()

        if cfg.dataset.name == 'BEHAVE':
            self.smpl = SMPLHLayer(model_path=cfg.model.smplh_dir, gender='male')
            self.dataset_metadata = BEHAVEMetaData(cfg.dataset.root_dir)
            pca_models_path = 'data/datasets/behave_pca_models_n{}_{}_d{}.pkl'.format(cfg.model.smpl_anchor_num, cfg.model.object_anchor_num, cfg.model.pca_dim)
        elif cfg.dataset.name == 'InterCap':
            self.smpl = SMPLXLayer(model_path=cfg.model.smplx_dir, gender='neutral')
            self.dataset_metadata = InterCapMetaData(cfg.dataset.root_dir)
            pca_models_path = 'data/datasets/intercap_pca_models_n{}_{}_d{}.pkl'.format(cfg.model.smpl_anchor_num, cfg.model.object_anchor_num, cfg.model.pca_dim)
        elif cfg.dataset.name == 'BEHAVE-Extended':
            self.smpl = SMPLHLayer(model_path=cfg.model.smplh_dir, gender='male')
            self.dataset_metadata = BEHAVEExtendMetaData(cfg.dataset.root_dir)
            pca_models_path = 'data/datasets/behave_extend_pca_models_n{}_{}_d{}.pkl'.format(cfg.model.smpl_anchor_num, cfg.model.object_anchor_num, cfg.model.pca_dim)

        obj_templates = self.dataset_metadata.obj_mesh_templates
        pca_models = load_pickle(pca_models_path)
        object_anchors = []
        pca_means = []
        pca_components = []
        for object_idx in sorted(self.dataset_metadata.OBJECT_IDX2NAME.keys()):
            object_name = self.dataset_metadata.OBJECT_IDX2NAME[object_idx]
            verts = obj_templates[object_name][0]
            anchors = verts[pca_models[object_name]['object_anchor_indices']]
            object_anchors.append(anchors)
            pca_means.append(pca_models[object_name]['mean'])
            pca_components.append(pca_models[object_name]['components'])
        object_anchors = torch.tensor(object_anchors, dtype=torch.float32)
        self.register_buffer('object_anchors', object_anchors)
        smpl_anchor_indices = torch.tensor(pca_models[object_name]['smpl_anchor_indices'], dtype=torch.int64).reshape(-1)
        self.register_buffer('smpl_anchor_indices', smpl_anchor_indices)
        pca_means = torch.tensor(pca_means, dtype=torch.float32)
        self.register_buffer('pca_means', pca_means)
        pca_components = torch.tensor(pca_components, dtype=torch.float32)
        self.register_buffer('pca_components', pca_components)


    def encode(self, smpl_betas, smpl_body_pose_rotmat, object_rel_rotmat, object_rel_trans, object_labels):
        b = smpl_betas.shape[0]
        smpl_out = self.smpl(body_pose=smpl_body_pose_rotmat, betas=smpl_betas)
        smpl_J = smpl_out.joints
        smpl_v = smpl_out.vertices
        smpl_v = smpl_v - smpl_J[:, :1]
        smpl_anchors = torch.stack([v[self.smpl_anchor_indices] for v in smpl_v], dim=0)

        object_anchors = self.object_anchors[object_labels]
        object_anchors = torch.matmul(object_anchors, object_rel_rotmat.permute(0, 2, 1)) + object_rel_trans.unsqueeze(1)
        offsets = object_anchors.unsqueeze(1) - smpl_anchors.unsqueeze(2)
        offsets = offsets.reshape(b, -1)

        gamma = torch.matmul(self.pca_components[object_labels], (offsets - self.pca_means[object_labels]).unsqueeze(-1)).squeeze(-1)

        return gamma, offsets


    def decode(self, gamma, object_labels):
        offsets = torch.matmul(self.pca_components[object_labels].permute(0, 2, 1), gamma.unsqueeze(-1)).squeeze(-1) + self.pca_means[object_labels]
        return offsets


    def decode_object_RT(self, offsets, smpl_betas, smpl_body_pose_rotmat, object_labels):
        b = smpl_betas.shape[0]
        smpl_out = self.smpl(body_pose=smpl_body_pose_rotmat, betas=smpl_betas)
        smpl_J = smpl_out.joints
        smpl_v = smpl_out.vertices
        smpl_v = smpl_v - smpl_J[:, :1]
        smpl_anchors = torch.stack([v[self.smpl_anchor_indices] for v in smpl_v], dim=0)

        object_anchors = self.object_anchors[object_labels]
        m, n = smpl_anchors.shape[1], object_anchors.shape[1]
        offsets = offsets.reshape(b, m, n, 3)
        smpl_anchors = smpl_anchors.reshape(b, m, 1, 3).repeat(1, 1, n, 1)
        object_p = smpl_anchors + offsets
        P = object_p.reshape(b, -1, 3)
        object_q = object_anchors.reshape(b, 1, n, 3).repeat(1, m, 1, 1)
        Q = object_q.reshape(b, -1, 3)
        center_Q = Q.mean(1).reshape(b, -1, 3)
        Q = Q - center_Q
        svd_mat = P.transpose(1, 2) @ Q
        svd_mat = svd_mat.double() # [b, 3, 3]
        u, _, v = torch.svd(svd_mat)
        d = torch.det(u @ v.transpose(1, 2)) # [b, ]
        d = torch.cat([
            torch.ones(b, 2, device=u.device),
            d.unsqueeze(-1)], axis=-1) # [b, 3]
        d = torch.eye(3, device=u.device).unsqueeze(0) * d.view(-1, 1, 3)
        obj_rotmat = u @ d @ v.transpose(1, 2)
        obj_rotmat_pred = obj_rotmat.to(object_q.dtype) # (b * n, 3, 3)
        _Q = Q + center_Q
        obj_trans_pred = (P.transpose(1, 2) - obj_rotmat_pred @ _Q.transpose(1, 2)).mean(dim=2) # (n * b, 3)
        object_rel_R = obj_rotmat_pred.reshape(b, 3, 3)
        object_rel_T = obj_trans_pred.reshape(b, 3)

        return object_rel_R, object_rel_T
