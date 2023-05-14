import os
import numpy as np
import torch
from smplx import SMPLH
from pytorch3d.transforms import axis_angle_to_matrix

from stackflow.datasets.behave_metadata import OBJECT_NAME2IDX, load_object_mesh_templates
from stackflow.datasets.data_utils import load_pickle


class BEHAVEHOITemplateDataset(object):

    def __init__(self, datalist_file, object_name='backpack', is_train=True):
        self.train = is_train
        self.smplh = SMPLH(model_path='data/smplh', gender='male', ext='pkl')

        data_list = load_pickle(datalist_file)
        _data_list = []
        all_image_ids = []
        for item in data_list:
            day_id, sub_id, obj_name, inter_type, frame_id, cam_id = item['img_id'].split('_')
            duplicated = False
            for cam_id_ in range(4):
                img_id_ = '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, str(cam_id_)])
                if img_id_ in all_image_ids:
                    duplicated = True
            if not duplicated and (object_name == 'all' or item['object_label'] == OBJECT_NAME2IDX[object_name]):
                all_image_ids.append(item['img_id'])
            # if int(cam_id) != 1 or object_name != 'all' and item['object_label'] != OBJECT_NAME2IDX[object_name]:
            #     continue
                _data_list.append(item)
        self.data_list = _data_list
        self.object_templates = load_object_mesh_templates()


    def __len__(self, ):
        return len(self.data_list)


    def __getitem__(self, idx):
        item = self.data_list[idx]
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = item['img_id'].split('_')

        smplh_betas = torch.tensor(item['smplh_betas'], dtype=torch.float32).reshape(1, -1)
        smplh_pose = torch.tensor(item['smplh_pose'], dtype=torch.float32).reshape(1, -1)
        smplh_trans = torch.tensor(item['smplh_trans'], dtype=torch.float32).reshape(1, -1)

        smplh_out = self.smplh(betas=smplh_betas, body_pose=smplh_pose[:, 3:66], global_orient=0*smplh_pose[:, :3], transl=0*smplh_trans)
        smplh_verts = smplh_out.vertices.detach().reshape(-1, 3)

        root_rotmat = axis_angle_to_matrix(smplh_pose[:, :3]).reshape(1, 3, 3)
        J = smplh_out.joints.detach()
        root_T = smplh_trans - (torch.einsum('bst,bvt->bvs', root_rotmat, J[:, 0:1]) - J[:, 0:1]).squeeze(1)
        root_T = root_T.reshape(3, )

        object_rotmat = torch.tensor(item['object_rotmat'], dtype=torch.float32).reshape(3, 3)
        object_trans = torch.tensor(item['object_trans'], dtype=torch.float32).reshape(3, )
        object_rotmat = root_rotmat[0].transpose(1, 0) @ object_rotmat
        object_trans = root_rotmat[0].transpose(1, 0) @ (object_trans - root_T).reshape(3, 1)

        object_verts_org, object_faces = self.object_templates[obj_name]
        max_points = 1700
        object_verts_org = torch.cat([object_verts_org, 
                                      torch.zeros(max_points - object_verts_org.shape[0], 3, dtype=object_verts_org.dtype)], dim=0)
        object_verts = torch.einsum('st,vt->vs', object_rotmat, object_verts_org) + object_trans.reshape(1, 3)

        return {
            'item_id': item['img_id'],
            'object_category': item['object_label'],
            'smplh_verts': smplh_verts,
            'object_verts': object_verts,
            'smpl_body_pose': smplh_pose[:, 3:72].reshape(69, ),
            'smpl_betas': smplh_betas.reshape(10, ),
            'object_R': object_rotmat,
            'object_T': object_trans,
        }
