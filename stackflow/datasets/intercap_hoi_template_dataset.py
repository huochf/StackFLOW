import os, sys
import numpy as np
import torch
import trimesh
from smplx import SMPLX
from scipy.spatial import cKDTree as KDTree
from pytorch3d.transforms import axis_angle_to_matrix

from stackflow.datasets.intercap_metadata import OBJECT_NAME2IDX, OBJECT_IDX2NAME, load_object_mesh_templates
from stackflow.datasets.data_utils import load_pickle


class InterCapHOITemplateDataset(object):

    def __init__(self, datalist_file, object_name='suitcase1', is_train=True):
        self.train = is_train
        self.smplx = SMPLX(model_path='data/smplx', gender='male', ext='pkl', num_pca_comps=12,).to('cuda')

        # self.smplx_parts = load_pickle('data/intercap/smplx_parts_dense.pkl')
        smplx_pkl = load_pickle('data/smplx/SMPLX_MALE.pkl')
        self.v_labels = np.argmax(smplx_pkl['weights'], axis=1)

        data_list = load_pickle(datalist_file)
        _data_list = []
        img_ids_all = {}
        for item in data_list:
            sub_id, obj_id, seg_id, cam_id, frame_id = item['img_id'].split('_')
            duplicated = False
            for cam_id_ in range(1, 7):
                img_id_ = '_'.join([sub_id, obj_id, seg_id, str(cam_id_), frame_id])
                if img_id_ in img_ids_all:
                    duplicated = True
            if not duplicated and (object_name == 'all' or item['object_label'] == OBJECT_NAME2IDX[object_name]):
                _data_list.append(item)
                img_ids_all[item['img_id']] = 1
        self.data_list = _data_list
        self.object_templates = load_object_mesh_templates()

        # self.hand_annotations = load_pickle('data/intercap/intercap_data_list_with_hand.pkl')
        # self.hand_annotations = {item['img_id']: item for item in self.hand_annotations}


    def __len__(self, ):
        return len(self.data_list)


    def __getitem__(self, idx):
        item = self.data_list[idx]
        sub_id, obj_id, seg_id, cam_id, frame_id = item['img_id'].split('_')
        smplh_betas = torch.tensor(item['smplh_betas'], dtype=torch.float32).reshape(1, -1).to('cuda')
        smplh_pose = torch.tensor(item['smplh_pose'], dtype=torch.float32).reshape(1, -1).to('cuda')
        smplh_trans = torch.tensor(item['smplh_trans'], dtype=torch.float32).reshape(1, -1).to('cuda')


        # left_hand_pose = torch.tensor(self.hand_annotations[item['img_id']]['left_hand_pose'], dtype=torch.float32).reshape(1, -1).to('cuda')
        # right_hand_pose = torch.tensor(self.hand_annotations[item['img_id']]['right_hand_pose'], dtype=torch.float32).reshape(1, -1).to('cuda')

        smplh_out = self.smplx(betas=smplh_betas, body_pose=smplh_pose[:, 3:66], global_orient=0*smplh_pose[:, :3], transl=0*smplh_trans,)
                               # left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
        smplh_verts = smplh_out.vertices.detach().reshape(-1, 3).cpu()

        root_rotmat = axis_angle_to_matrix(smplh_pose[:, :3]).reshape(1, 3, 3).cpu()
        J = smplh_out.joints.detach().cpu()
        root_T = smplh_trans.cpu() - (torch.einsum('bst,bvt->bvs', root_rotmat, J[:, 0:1]) - J[:, 0:1]).squeeze(1)
        root_T = root_T.reshape(3, )

        object_rotmat = torch.tensor(item['object_rotmat'], dtype=torch.float32).reshape(3, 3)
        object_trans = torch.tensor(item['object_trans'], dtype=torch.float32).reshape(3, )
        object_rotmat = root_rotmat[0].transpose(1, 0) @ object_rotmat
        object_trans = root_rotmat[0].transpose(1, 0) @ (object_trans - root_T).reshape(3, 1)

        object_verts_org, object_faces = self.object_templates[OBJECT_IDX2NAME[int(obj_id) - 1]]
        max_points = 2500
        object_verts_org = torch.cat([object_verts_org, 
                                      torch.zeros(max_points - object_verts_org.shape[0], 3, dtype=object_verts_org.dtype)], dim=0)
        object_verts = torch.einsum('st,vt->vs', object_rotmat, object_verts_org) + object_trans.reshape(1, 3)

        # person_pc = smplh_verts.numpy()
        # left_hand_verts = person_pc[self.smplx_parts['left_forearm']]
        # right_hand_verts = person_pc[self.smplx_parts['right_forearm']]
        # sys.stdout.flush()
        # object_pc = object_verts.numpy()
        # kdtree = KDTree(object_pc)

        # dist, idx = kdtree.query(left_hand_verts)
        # left_in_contact = dist.min() < 0.05

        # dist, idx = kdtree.query(right_hand_verts)
        # right_in_contact = dist.min() < 0.05

        # trimesh.Trimesh(vertices=person_pc, faces=self.smplx.faces, process=False).export('./debug_smpl.ply')
        # trimesh.Trimesh(vertices=object_pc[:len(object_verts_org)], faces=object_faces.numpy(), process=False).export('./debug_object.ply')

        return {
            'item_id': item['img_id'],
            'object_category': item['object_label'],
            'smplh_verts': smplh_verts,
            'object_verts': object_verts,
            'smpl_body_pose': smplh_pose[:, 3:].reshape(63, ),
            'smpl_betas': smplh_betas.reshape(10, ),
            'object_R': object_rotmat,
            'object_T': object_trans,
            # 'left_in_contact': left_in_contact,
            # 'right_in_contact': right_in_contact,
        }
