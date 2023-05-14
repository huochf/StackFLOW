import os
import numpy as np
import torch
import torch.nn as nn

from stackflow.datasets.behave_metadata import OBJECT_NAME2IDX as BEHAVE_OBJECT_NAME2IDX
from stackflow.datasets.intercap_metadata import OBJECT_NAME2IDX as INTERCAP_OBJECT_NAME2IDX
from stackflow.datasets.data_utils import load_pickle


class DistanceDecoder(nn.Module):

    def __init__(self, params_dir, anchor_num, pca_dim=32):
        super(DistanceDecoder, self).__init__()
        pca_models = load_pickle(params_dir)
        # pca_models = load_pickle(os.path.join(params_dir, 'pca_models_n{}_d{}.pkl'.format(anchor_num, pca_dim)))

        if 'behave' in params_dir:
            OBJECT_NAME2IDX = BEHAVE_OBJECT_NAME2IDX
        else:
            OBJECT_NAME2IDX = INTERCAP_OBJECT_NAME2IDX

        object_anchor_indices = []
        means, components = [], []
        for obj_name in OBJECT_NAME2IDX.keys():
            obj_indices = pca_models[obj_name]['object_anchor_indices']
            object_anchor_indices.append(obj_indices)

            smpl_anchor_indices = pca_models[obj_name]['smpl_anchor_indices']
            means.append(pca_models[obj_name]['mean'])
            components.append(pca_models[obj_name]['components'])
        self.register_buffer('object_anchor_indices', torch.tensor(np.array(object_anchor_indices), dtype=torch.int64)) # (num_obj, num_anchors)
        self.register_buffer('smpl_anchor_indices', torch.tensor(np.array(smpl_anchor_indices), dtype=torch.int64)) # (num_acnhors)
        self.register_buffer('means', torch.tensor(np.array(means), dtype=torch.float32)) # (num_objects, n)
        self.register_buffer('components', torch.tensor(np.array(components), dtype=torch.float32)) # (num_objects, d, n)


    def forward(self, lattent_codes, object_labels):
        b = lattent_codes.shape[0]
        means = self.means[object_labels]
        components = self.components[object_labels]
        rel_dist = torch.matmul(lattent_codes.unsqueeze(1), components).squeeze(1) + means

        return rel_dist


    def encode(self, rel_dist, object_labels):
        b = rel_dist.shape[0]
        means = self.means[object_labels]
        components = self.components[object_labels]
        lattent_codes = torch.matmul((rel_dist - means).unsqueeze(1), components.permute(0, 2, 1)).squeeze(1)
        return lattent_codes


    def get_rel_dist(self, smpl_v, object_v, object_labels):
        b = len(object_labels)
        object_indices = self.object_anchor_indices[object_labels]
        object_anchors = torch.stack([object_v[i, indices] for i, indices in enumerate(object_indices)], dim=0) # (b, m, 3)
        person_anchors = torch.stack([smpl_v[i, self.smpl_anchor_indices] for i in range(b)], dim=0)
        rel_dist = object_anchors.reshape(b, 1, -1, 3) - person_anchors.reshape(b, -1, 1, 3)
        rel_dist = rel_dist.reshape(b, -1)

        return rel_dist
