import os
import sys
import json
import trimesh
import math
import cv2
import numpy as np
from typing import Dict
from easydict import EasyDict
import yaml
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from smplx import SMPLHLayer, SMPLXLayer
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss.point_mesh_distance import point_face_distance

from stackflow.models.model import Model

from stackflow.datasets.behave_hoi_img_dataset import BEHAVEImageDataset
from stackflow.datasets.behave_metadata import MAX_OBJECT_VERT_NUM as BEHAVE_MAX_OBJECT_VERT_NUM
from stackflow.datasets.intercap_hoi_img_dataset import IntercapImageDataset
from stackflow.datasets.behave_metadata import get_gt_mesh as get_behave_gt_mesh
from stackflow.datasets.intercap_metadata import get_gt_mesh as get_intercap_gt_mesh
from stackflow.datasets.behave_metadata import OBJECT_IDX2NAME as BEHAVE_OBJECT_IDX2NAME
from stackflow.datasets.behave_metadata import load_object_mesh_templates as load_behave_object_mesh_templates
from stackflow.datasets.intercap_metadata import MAX_OBJECT_VERT_NUM as INTERCAP_MAX_OBJECT_VERT_NUM
from stackflow.datasets.intercap_metadata import OBJECT_IDX2NAME as INTERCAP_OBJECT_IDX2NAME
from stackflow.datasets.intercap_metadata import load_object_mesh_templates as load_intercap_object_mesh_templates
from stackflow.datasets.data_utils import save_json, to_device
from stackflow.utils.evaluator import MeshEvaluator, ReconEvaluator
from stackflow.utils.optimization import joint_optimize
from stackflow.utils.optimize_loss import SMPLPriorLoss, RelativeDistanceLoss, PersonKeypointLoss, ObjectReprojLoss
from stackflow.utils.optimize_loss import SMPLPostperioriLoss, RelDistPostperioriLoss, RelativeDistanceFlowLoss
from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds


def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
            lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    return cham_x, cham_y


class HOIInstance(nn.Module):

    def __init__(self, hoi_flow, object_labels, condition_features, smpl_trans=None,
        object_T=None, object_R=None, smpl_type='smplh', dataset='behave', fix_person=False, device=torch.device('cuda')):
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
            
        if object_T is not None:
            self.object_T = nn.Parameter(object_T.reshape(batch_size, 3))
        else:
            self.object_T = nn.Parameter(self.smpl_trans.clone())

        if object_R is not None:
            self.object_R6d = nn.Parameter(matrix_to_rotation_6d(object_R.reshape(batch_size, 3, 3)))
        else:
            self.object_R6d = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)))

        self.fix_person = fix_person
        self.optimizer = None


    def get_optimizer(self, fix_smpl_trans=False, fix_smpl_global_orient=False, lr=0.001):
        param_list = [self.smpl_z, self.object_T, self.object_R6d]
        if not fix_smpl_trans:
            param_list.append(self.smpl_trans)

        if self.fix_person:
            param_list = [self.object_T, self.object_R6d]

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

        object_v = self.object_v
        object_v = compute_transformation_persp(object_v, self.object_T.reshape(batch_size, 1, 3), self.object_R6d)

        results = {
            "smpl_v": smpl_out.vertices,
            "object_v": object_v,
            'smpl_betas': betas,
            'smpl_body_pose': matrix_to_axis_angle(pose_rotmat[:, 1:22]).reshape(batch_size, 63),
            'smpl_trans': self.smpl_trans,
            'smpl_global_orient': matrix_to_axis_angle(pose_rotmat[:, 0]),
            'object_R': rotation_6d_to_matrix(self.object_R6d),
            'object_T': self.object_T,
            'object_global_rotmat': rotation_6d_to_matrix(self.object_R6d),
            'object_global_trans': self.object_T,
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


def point_to_face_distance(meshes: Meshes, pcls: Pointclouds, min_triangle_area: float = 5e-3):
    if len(meshes) != len(pcls):
        raise ValueError('meshes and pointclouds must be equal sized batches')
    N = len(meshes)

    # packes representation for pointclouds
    points = pcls.points_packed() # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed] # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    point_to_face = point_face_distance(points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area)

    return point_to_face


class HOIContactLoss(nn.Module):

    def __init__(self, smpl_contact_maps, object_contact_maps, smpl_faces, object_faces, object_verts_n):
        super(HOIContactLoss, self).__init__()
        self.smpl_contact_maps = smpl_contact_maps.cuda()
        self.object_contact_maps = object_contact_maps.cuda()
        self.smpl_faces = smpl_faces.cuda().unsqueeze(0)
        self.object_faces = [f.cuda() for f in object_faces]
        self.object_verts_n = object_verts_n


    def forward(self, hoi_dict):
        smpl_v = hoi_dict['smpl_v']
        object_v = hoi_dict['object_v']
        contact_loss = 0
        no_contact_loss = 0
        batch_size = smpl_v.shape[0]
        for i in range(batch_size):
            smpl_d, object_d = chamfer_distance(smpl_v[i].unsqueeze(0), object_v[i][:self.object_verts_n[i]].unsqueeze(0))
            smpl_contact_loss = self.smpl_contact_maps[i].squeeze(1) * smpl_d.squeeze(0)
            object_contact_loss = self.object_contact_maps[i][:self.object_verts_n[i]].squeeze(1) * object_d.squeeze(0)
            # contact_loss += smpl_contact_loss.mean() + object_contact_loss.mean()
            contact_loss += smpl_contact_loss.sum() / (self.smpl_contact_maps[i].sum() + 1e-6) + object_contact_loss.sum() / (self.object_contact_maps[i][:self.object_verts_n[i]].sum() + 1e-6)

            # smpl_mesh = Meshes(smpl_v[i].unsqueeze(0), self.smpl_faces)
            # object_mesh = Meshes(object_v[i][:self.object_verts_n[i]].unsqueeze(0), self.object_faces[i].unsqueeze(0))
            # smpl_pc = Pointclouds(points=smpl_mesh.verts_list())
            # object_pc = Pointclouds(points=object_mesh.verts_list())
            # smpl_contact_loss = self.smpl_contact_maps[i] * point_to_face_distance(object_mesh, Pointclouds(points=smpl_mesh.verts_list()))
            # smpl_contact_loss = smpl_contact_loss.sum() / self.smpl_contact_maps[i].sum()
            # object_contact_loss = self.object_contact_maps[i][:self.object_verts_n[i]] * point_to_face_distance(smpl_mesh, Pointclouds(points=object_mesh.verts_list()))
            # object_contact_loss = object_contact_loss.sum() / self.object_contact_maps[i][:self.object_verts_n[i]].sum()
            # contact_loss += smpl_contact_loss + object_contact_loss

        return {
            'hoi_contact_loss': contact_loss / batch_size,
        }


def main(cfg):

    device = torch.device('cuda')

    if cfg.dataset.name == 'behave':
        test_dataset = BEHAVEImageDataset(cfg, exlusion_occlusion=True, is_train=False)
        print(len(test_dataset))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, num_workers=4, shuffle=True, drop_last=False) # cfg.train.batch_size
        get_gt_mesh = get_behave_gt_mesh
        object_templates = load_behave_object_mesh_templates()
        OBJECT_IDX2NAME = BEHAVE_OBJECT_IDX2NAME
    else:
        test_dataset = IntercapImageDataset(cfg, exlusion_occlusion=True, is_train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, drop_last=False) # cfg.train.batch_size
        get_gt_mesh = get_intercap_gt_mesh
        object_templates = load_intercap_object_mesh_templates()
        OBJECT_IDX2NAME = INTERCAP_OBJECT_IDX2NAME

    model = Model(cfg)
    model.to(device)

    checkpoint_path = os.path.join(cfg.train.checkpoint_out_dir, '{}.pth'.format(cfg.train.exp))
    if not os.path.exists(checkpoint_path):
        print('Cannot found checkpoints: {}.'.format(checkpoint_path))
        return
    model.load_checkpoint(checkpoint_path)
    model.eval()
    print('Loaded model from {}.'.format(checkpoint_path))

    evaluator = MeshEvaluator()
    evaluator_chore = ReconEvaluator(smpl_only=False)

    if cfg.dataset.name == 'behave':
        person_keypoints_path = 'data/behave/person_keypoints_test.json'
        with open(person_keypoints_path, 'r') as f:
            person_keypoints = json.load(f)
        object_coor_path = 'data/behave/epro_pnp_behave_recon_results.pkl'
        with open(object_coor_path, 'rb') as f:
            object_coor = pickle.load(f)
        hoi_contact_path = 'outputs/bstro_hoi/behave_bstro_hoi_outputs.pkl'
        with open(hoi_contact_path, 'rb') as f:
            hoi_contact = pickle.load(f)
    else:
        person_keypoints_path = 'data/intercap/person_keypoints_test.json'
        with open(person_keypoints_path, 'r') as f:
            person_keypoints = json.load(f)
        object_coor_path = 'data/intercap/epro_pnp_intercap_recon_results.pkl'
        with open(object_coor_path, 'rb') as f:
            object_coor = pickle.load(f)
        hoi_contact_path = 'outputs/bstro_hoi/intercap_bstro_hoi_outputs.pkl'
        with open(hoi_contact_path, 'rb') as f:
            hoi_contact = pickle.load(f)

    eval_metrics = {}
    recon_results = {}
    count = 0
    # from timeit import default_timer as timer
    # begin_time = timer()
    for idx, batch in enumerate(test_dataloader):
        batch = to_device(batch, device)

        outputs = model.inference_step(batch)
        relative_distance = outputs['rel_dist_recon'][:, 0]
        smpl_trans = outputs['translation']
        object_labels = batch['object_label']

        img_ids = batch['img_id']
        try:
            keypoints = torch.from_numpy(np.array([person_keypoints[img_id]['keypoints'] for img_id in img_ids], dtype=np.float32))
            confidences = torch.from_numpy(np.array([person_keypoints[img_id]['confidence'] for img_id in img_ids], dtype=np.float32))

            obj_x2d = torch.from_numpy(np.array([object_coor[img_id]['x2d'] for img_id in img_ids], dtype=np.float32))
            obj_x3d = torch.from_numpy(np.array([object_coor[img_id]['x3d'] for img_id in img_ids], dtype=np.float32))
            obj_w2d = torch.from_numpy(np.array([object_coor[img_id]['w2d'] for img_id in img_ids], dtype=np.float32))
            object_T = torch.from_numpy(np.array([np.array(object_coor[img_id]['pose_est'])[:, 3] for img_id in img_ids], dtype=np.float32))
            object_R = torch.from_numpy(np.array([np.array(object_coor[img_id]['pose_est'])[:, :3] for img_id in img_ids], dtype=np.float32))
            smpl_contact_maps = torch.from_numpy(np.array([hoi_contact[img_id]['smpl_full'] for img_id in img_ids], dtype=np.float32))
            object_contact_maps = torch.from_numpy(np.array([hoi_contact[img_id]['object_full'] for img_id in img_ids], dtype=np.float32))
        except:
            continue
        smpl_contact_maps[smpl_contact_maps < 0.5] = 0
        object_contact_maps[object_contact_maps < 0.5] = 0

        object_faces = []
        object_verts_n = []
        batch_size = batch['object_label'].shape[0]
        for i in range(batch_size):
            verts, faces = object_templates[OBJECT_IDX2NAME[batch['object_label'][i].item()]]
            object_faces.append(faces)
            object_verts_n.append(verts.shape[0])

        try:
            hoi_instance = HOIInstance(hoi_flow=model.hoi_flow, 
                                       object_labels=object_labels, 
                                       condition_features=outputs['conditioning_feats'],
                                       smpl_trans=smpl_trans, 
                                       object_T=object_T, 
                                       object_R=object_R,
                                       smpl_type='smplh' if cfg.dataset.name == 'behave' else 'smplx', 
                                       dataset=cfg.dataset.name, 
                                       fix_person=False, 
                                       device=device)
            hoi_instance.to(device)

            loss_functions = [
                PersonKeypointLoss(keypoints, confidences, batch['K'], smpl_type='smplh' if cfg.dataset.name == 'behave' else 'smplx').to(device),
                ObjectReprojLoss(obj_x3d, obj_x2d, obj_w2d, batch['K'], rescaled=False).to(device),
                SMPLPostperioriLoss().to(device),
                HOIContactLoss(smpl_contact_maps, object_contact_maps, torch.tensor(hoi_instance.smpl.faces.astype(np.int64)), object_faces, object_verts_n).to(device),
            ]
            loss_weights = {
                'object_reproj_loss': lambda cst, it: 10. ** 0 * cst / (1 + it),
                'person_keypoints_reproj_loss': lambda cst, it: 10. ** -1 * cst / (1 + it),
                'smpl_postperiori_loss': lambda cst, it: 10. ** 0 * cst / (1 + it),
                'hoi_contact_loss': lambda cst, it: 10. ** 0 * cst / (1 + it),
            }

            joint_optimize(hoi_instance, loss_functions, loss_weights, iterations=2, steps_per_iter=200, lr=0.01)

            hoi_instance_out = hoi_instance()
            batch_size = relative_distance.shape[0]
            for b in range(batch_size):
                gt_smpl_mesh, gt_object_mesh = get_gt_mesh(batch['img_id'][b])
                gt_smpl_vertices = np.array(gt_smpl_mesh.vertices)
                gt_object_vertices = np.array(gt_object_mesh.vertices)
                num_obj_v = gt_object_vertices.shape[0]
                recon_smpl_vertices = hoi_instance_out['smpl_v'][b].detach().cpu().numpy()
                recon_object_vertices = hoi_instance_out['object_v'][b].detach().cpu().numpy()[:num_obj_v]

                smpl_metrics = evaluator.eval_pointcloud(recon_smpl_vertices, gt_smpl_vertices)
                object_metrics = evaluator.eval_pointcloud(recon_object_vertices, gt_object_vertices)
                smpl_chamfer_distance, object_chamfer_distance = evaluator_chore.compute_errors([gt_smpl_mesh, gt_object_mesh], 
                    [trimesh.Trimesh(recon_smpl_vertices, gt_smpl_mesh.faces, process=False), trimesh.Trimesh(recon_object_vertices, gt_object_mesh.faces, process=False)])

                smpl_metrics['chamfer_distance_chore'] = smpl_chamfer_distance
                object_metrics['chamfer_distance_chore'] = object_chamfer_distance
                print('Joint Optim: {} / {}'.format(idx, len(test_dataloader)), batch['img_id'][b], smpl_metrics['chamfer-L1'], object_metrics['chamfer-L1'], smpl_chamfer_distance, object_chamfer_distance)
                sys.stdout.flush()
                eval_metrics[batch['img_id'][b]] = {
                    'object_joint_optim': object_metrics,
                    'smpl_joint_optim': smpl_metrics,
                }

                recon_results[batch['img_id'][b]] = {
                    'smpl_betas': hoi_instance_out['smpl_betas'][b].detach().cpu().numpy().tolist(),
                    'smpl_body_pose': hoi_instance_out['smpl_body_pose'][b].detach().cpu().numpy().tolist(),
                    'smpl_global_orient': hoi_instance_out['smpl_global_orient'][b].detach().cpu().numpy().tolist(),
                    'smpl_transl': hoi_instance_out['smpl_trans'][b].detach().cpu().numpy().tolist(),
                    'object_R': hoi_instance_out['object_R'][b].detach().cpu().numpy().tolist(),
                    'object_T': hoi_instance_out['object_T'][b].detach().cpu().numpy().tolist(),
                }
        except:
            continue

    avg_dict = {}
    for img_id, metric_dict in eval_metrics.items():
        for method, metric in metric_dict.items():
            if method not in avg_dict:  
                avg_dict[method] = {}
            for k, v in metric.items():
                if k not in avg_dict[method]:
                    avg_dict[method][k] = v / len(eval_metrics)
                else:
                    avg_dict[method][k] += v / len(eval_metrics)

    eval_metrics['avg'] = avg_dict
    print(avg_dict)
    with open(os.path.join(cfg.train.checkpoint_out_dir, '{}_bstro_optim_eval_metric_chore_occlusion.json'.format(cfg.train.exp)), 'w') as f:
        json.dump(eval_metrics, f, indent=4)
    with open(os.path.join(cfg.train.checkpoint_out_dir, '{}_bstro_optim_recon_results_chore_occlusion.json'.format(cfg.train.exp)), 'w') as f:
        json.dump(recon_results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/prohmr_behave.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    main(cfg)
