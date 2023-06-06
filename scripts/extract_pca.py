import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import numpy as np
import random
import trimesh
import argparse
from tqdm import tqdm
import torch

from smplx import SMPLHLayer, SMPLXLayer
from sklearn.decomposition import PCA

from stackflow.datasets.behave_metadata import BEHAVEMetaData
from stackflow.datasets.intercap_metadata import InterCapMetaData
from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from stackflow.datasets.utils import load_pickle, save_pickle


class HOOffsetDataList():

    def __init__(self, annotation_file, dataset_root, anchor_indices, obj_name, args):
        print('loading annotations ...')
        if isinstance(annotation_file, str):
            self.annotations = load_pickle(annotation_file)
        else:
            self.annotations = annotation_file
        self.is_behave = args.is_behave
        self.behave_extend = args.behave_extend
        self.anchor_indices = anchor_indices
        self.obj_name = obj_name
        if args.is_behave:
            self.dataset_metadata = BEHAVEMetaData(dataset_root)
            self.smpl = SMPLHLayer(model_path='data/models/smplh', gender='male', ext='pkl')
            self.annotations = [item for item in self.annotations if item['img_id'].split('_')[2] == obj_name]
        elif args.behave_extend:
            self.dataset_metadata = BEHAVEExtendMetaData(dataset_root)
            self.smpl = SMPLHLayer(model_path='data/models/smplh', gender='male', ext='pkl')
            self.annotations = [item for item in self.annotations if item['img_id'].split('_')[2] == obj_name]
        else:
            self.dataset_metadata = InterCapMetaData(dataset_root)
            self.smpl = SMPLXLayer(model_path='data/models/smplx', gender='neutral', ext='pkl')
            self.annotations = [item for item in self.annotations 
                if self.dataset_metadata.OBJECT_IDX2NAME[item['img_id'].split('_')[1]] == obj_name]
        random.shuffle(self.annotations)
        self.annotations = self.annotations[:30000] 
        # self.annotations = self.annotations[:100]# for debug

    def __len__(self, ):
        return len(self.annotations)


    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_id = annotation['img_id']
        if not self.is_behave and not self.behave_extend:
            obj_id = img_id.split('_')[1]
            obj_name = self.dataset_metadata.OBJECT_IDX2NAME[obj_id]
            object_v = self.dataset_metadata.obj_mesh_templates[obj_name][0]
            person_beta = torch.tensor(annotation['smplx_betas_neutral'].astype(np.float32)).reshape(1, 10)
            person_body_pose = torch.tensor(annotation['smplx_pose_rotmat'].astype(np.float32)[1:22]).reshape(1, 21, 3, 3)
            person_transl = torch.tensor(annotation['smplx_trans'].astype(np.float32)).reshape(1, 3)
            person_global_orient = torch.tensor(annotation['smplx_pose_rotmat'].astype(np.float32)[0]).reshape(1, 1, 3, 3)
        else:
            obj_name = img_id.split('_')[2]
            object_v = self.dataset_metadata.obj_mesh_templates[obj_name][0]
            person_beta = torch.tensor(annotation['smplh_betas_male'].astype(np.float32)).reshape(1, 10)
            person_body_pose = torch.tensor(annotation['smplh_pose_rotmat'].astype(np.float32)[1:22]).reshape(1, 21, 3, 3)
            person_transl = torch.tensor(annotation['smplh_trans'].astype(np.float32)).reshape(1, 3)
            person_global_orient = torch.tensor(annotation['smplh_pose_rotmat'].astype(np.float32)[0]).reshape(1, 1, 3, 3)

        person_global_orient = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3)
        smpl_out = self.smpl(betas=person_beta, body_pose=person_body_pose, global_orient=person_global_orient, transl=0 * person_transl,)

        smpl_v = smpl_out.vertices.detach().reshape(-1, 3).numpy()
        J_0 = smpl_out.joints.detach()[:, 0].reshape(1, 3).numpy()
        smpl_v = smpl_v - J_0

        object_r = annotation['object_rel_rotmat']
        object_t = annotation['object_rel_trans']
        object_v = np.matmul(object_v, object_r.T) + object_t.reshape((1, 3))

        smpl_indices = self.anchor_indices['smpl']
        object_indices = self.anchor_indices['object'][obj_name]

        smpl_anchors = smpl_v[smpl_indices]
        object_anchors = object_v[object_indices]

        offsets = object_anchors.reshape(1, -1, 3) - smpl_anchors.reshape(-1, 1, 3)
        offsets = offsets.reshape(-1)

        if np.isnan(offsets).any():
            return self.__getitem__(np.random.randint(len(self)))
        return offsets


def sample_smpl_object_anchors(args):

    if args.is_behave:
        dataset_metadata = BEHAVEMetaData(args.root_dir)
        smpl_pkl = load_pickle('data/models/smplh/SMPLH_MALE.pkl')
        radius = 0.02
    elif args.behave_extend:
        dataset_metadata = BEHAVEExtendMetaData(args.root_dir)
        smpl_pkl = load_pickle('data/models/smplh/SMPLH_MALE.pkl')
        radius = 0.02
    else:
        dataset_metadata = InterCapMetaData(args.root_dir)
        smpl_pkl = load_pickle('data/models/smplx/SMPLX_NEUTRAL.pkl')
        radius = 0.01

    weights = smpl_pkl['weights']
    parts_indices = np.argmax(weights[:, :22], axis=1)

    smpl_anchor_indices = []
    for i in range(22):
        part_anchor_indices = np.where(parts_indices == i)[0]
        part_anchors = part_anchor_indices[np.random.choice(len(part_anchor_indices), args.smpl_anchor_num)]
        smpl_anchor_indices.append(part_anchors.tolist())

    object_anchor_indices = {}
    object_templates = dataset_metadata.load_object_trimesh()
    for object_name, mesh in object_templates.items():
        _, face_index = trimesh.sample.sample_surface_even(mesh, count=args.object_anchor_num, radius=radius)
        while face_index.shape[0] < args.object_anchor_num:
            print('Try again.')
            _, face_index = trimesh.sample.sample_surface_even(mesh, count=args.object_anchor_num, radius=radius)

        vertex_indices = np.array(mesh.faces)[face_index][:, 0]
        object_anchor_indices[object_name] = vertex_indices.tolist()

    anchor_indices = {'smpl': smpl_anchor_indices, 'object': object_anchor_indices}
    return anchor_indices


def extract_pca_models(args, anchor_indices):

    if args.is_behave:
        dataset_metadata = BEHAVEMetaData(args.root_dir)
        annotation_file = './data/datasets/behave_train_list.pkl'
    elif args.behave_extend:
        dataset_metadata = BEHAVEExtendMetaData(args.root_dir)
        annotation_file = './data/datasets/behave_extend_train_list.pkl'
        annotation_file = load_pickle(annotation_file)
    else:
        dataset_metadata = InterCapMetaData(args.root_dir)
        annotation_file = './data/datasets/intercap_train_list.pkl'

    pca_models = {}
    obj_names = list(dataset_metadata.OBJECT_NAME2IDX.keys())
    for obj_name in obj_names:
        dataset = HOOffsetDataList(annotation_file, args.root_dir, anchor_indices, obj_name, args)
        all_offsets = []
        print('collect offsets for object: {}'.format(obj_name))
        for offsets in tqdm(dataset, total=len(dataset)):
            all_offsets.append(offsets)
        pca = PCA(n_components=args.pca_dim)
        if len(all_offsets) == 0:
            print('no annotations for {} are found ...'.format(obj_name))
            n_offsets = args.smpl_anchor_num * 22 * args.object_anchor_num * 3
            pca_models[obj_name] = {
                'mean': np.zeros(n_offsets),
                'components': np.zeros((args.pca_dim, n_offsets)),
                'smpl_anchor_indices': anchor_indices['smpl'],
                'object_anchor_indices': anchor_indices['object'][obj_name]
            }
        else:
            all_offsets = np.stack(all_offsets, axis=0)
            print('principle components analysing ...')
            pca.fit_transform(all_offsets)
            pca_models[obj_name] = {
                'mean': pca.mean_,
                'components': pca.components_,
                'smpl_anchor_indices': anchor_indices['smpl'],
                'object_anchor_indices': anchor_indices['object'][obj_name]
            }
    return pca_models


def evaluate_pca_models(args, anchor_indices, pca_models):
    if args.is_behave:
        dataset_metadata = BEHAVEMetaData(args.root_dir)
        annotation_file = './data/datasets/behave_test_list.pkl'
    elif args.behave_extend:
        dataset_metadata = BEHAVEExtendMetaData(args.root_dir)
        annotation_file = './data/datasets/behave_extend_test_list.pkl'
        annotation_file = load_pickle(annotation_file)
    else:
        dataset_metadata = InterCapMetaData(args.root_dir)
        annotation_file = './data/datasets/intercap_test_list.pkl'

    all_lattent_code = []
    for obj_name in pca_models:
        pca_model = pca_models[obj_name]
        mean = pca_model['mean']
        components = pca_model['components']
        print('evaluate pca model for object: {}'.format(obj_name))
        dataset = HOOffsetDataList(annotation_file, args.root_dir, anchor_indices, obj_name, args)
        reconstruction_error = []
        eval_number = min(200, len(dataset))
        for i in tqdm(range(eval_number), total=eval_number):
            offset = dataset[i]

            latent_code = np.matmul(components, offset - mean)
            recon_offset = np.matmul(components.T, latent_code) + mean
            recon_error = np.abs(offset - recon_offset).mean()
            reconstruction_error.append(recon_error)
            all_lattent_code.append(latent_code)
        print('reconstruction error: {} +- {}'.format(np.mean(reconstruction_error), np.std(reconstruction_error)))
        print(np.array(all_lattent_code).mean(0))
    print(np.array(all_lattent_code).mean(0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str, help='Dataset root directory.')
    parser.add_argument('--is_behave', default=False, action="store_true")
    parser.add_argument('--behave_extend', default=False, action="store_true")
    parser.add_argument('--smpl_anchor_num', default=32, type=int, help='the number of SMPL anchors per body part')
    parser.add_argument('--object_anchor_num', default=64, type=int, help='the number of object anchors')
    parser.add_argument('--pca_dim', default=32, type=int, help='the number of dimensions of PCA latent space')

    args = parser.parse_args()

    np.random.seed(7) # for reproducibility
    random.seed(7)

    anchor_indices = sample_smpl_object_anchors(args)
    pca_models = extract_pca_models(args, anchor_indices)
    if args.is_behave:
        out_path = 'data/datasets/behave_pca_models_n{}_{}_d{}.pkl'.format(args.smpl_anchor_num, args.object_anchor_num, args.pca_dim)
    elif args.behave_extend:
        out_path = 'data/datasets/behave_extend_pca_models_n{}_{}_d{}.pkl'.format(args.smpl_anchor_num, args.object_anchor_num, args.pca_dim)
    else:
        out_path = 'data/datasets/intercap_pca_models_n{}_{}_d{}.pkl'.format(args.smpl_anchor_num, args.object_anchor_num, args.pca_dim)
    save_pickle(pca_models, out_path)
    evaluate_pca_models(args, anchor_indices, pca_models)
