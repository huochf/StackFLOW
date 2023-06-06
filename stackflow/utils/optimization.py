import numpy as np
import torch
from tqdm import tqdm

from smplx import SMPLHLayer, SMPLXLayer

from stackflow.models.hoi_instances import HOIInstance
from stackflow.datasets.utils import load_J_regressor
from stackflow.utils.optim_losses import ObjectReprojLoss, PersonKeypointLoss, PosteriorLoss


def get_behave_loss_weights():
    # For BEHAVE
    return {
        'person_reproj_loss': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
        'object_reproj_loss':  lambda cst, it: 10. ** 1 * cst / (1 + 10 * it),
        'smpl_pose_posterior_loss': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
        'offset_posterior_loss':  lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
    }


def get_intercap_loss_weights():
    return {
        'person_reproj_loss': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
        'object_reproj_loss':  lambda cst, it: 10. ** 1 * cst / (1 + 10 * it),
        'smpl_pose_posterior_loss': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
        'offset_posterior_loss':  lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
    }


def get_confidence_weights(offset_posterior_loss, scale=.2):
    weights = (offset_posterior_loss) * scale
    weights = 4 * (- weights).exp()
    return weights


def post_optimization(cfg, dataset_metadata, model, batch, predictions):
    device = torch.device('cuda')

    if cfg.dataset.name == 'BEHAVE' or cfg.dataset.name == 'BEHAVE-Extended':
        smpl = SMPLHLayer(model_path=cfg.model.smplh_dir, gender='male').to(device)
        J_regressor = load_J_regressor(cfg.model.smplh_regressor)
        loss_weights = get_behave_loss_weights()
    elif cfg.dataset.name == 'InterCap':
        smpl = SMPLXLayer(model_path=cfg.model.smplx_dir, gender='neutral').to(device)
        J_regressor = load_J_regressor(cfg.model.smplx_regressor)
        loss_weights = get_intercap_loss_weights()
    else:
        assert False

    num_object = len(dataset_metadata.OBJECT_IDX2NAME)
    object_v = np.zeros((num_object, dataset_metadata.object_max_vertices_num, 3))
    for idx, object_idx in enumerate(sorted(dataset_metadata.OBJECT_IDX2NAME.keys())):
        object_name = dataset_metadata.OBJECT_IDX2NAME[object_idx]
        vertices = dataset_metadata.obj_mesh_templates[object_name][0]
        object_v[idx, :len(vertices)] = vertices
    object_v = torch.tensor(object_v, dtype=torch.float32)
    object_labels = batch['object_labels'].cpu()
    object_v = object_v[object_labels]

    hoi_instance = HOIInstance(smpl=smpl, 
                               object_v=object_v, 
                               J_regressor=J_regressor,
                               smpl_betas=predictions['pred_betas'].detach(), 
                               smpl_body_pose6d=predictions['pred_pose6d'].detach()[:, 1:], 
                               obj_rel_trans=predictions['pred_obj_rel_T'].detach(), 
                               obj_rel_rotmat=predictions['pred_obj_rel_R'].detach(), 
                               hoi_trans=predictions['hoi_trans'].detach(), 
                               hoi_rot6d=predictions['pred_pose6d'].detach()[:, 0]).to(device)
    optimizer = hoi_instance.get_optimizer(fix_trans=False, fix_global_orient=False, lr=cfg.eval.optim_lr)

    loss_functions = [
        ObjectReprojLoss(model_points=batch['obj_x3d'], 
                         image_points=batch['obj_x2d'], 
                         pts_confidence=batch['obj_w2d'], 
                         focal_length=batch['focal_length'], 
                         optical_center=batch['optical_center'],).to(device),
        PersonKeypointLoss(keypoints=batch['person_kps'][:, :, :-1], 
                           confidences=batch['person_kps'][:, :, -1:],
                           focal_length=batch['focal_length'], 
                           optical_center=batch['optical_center']).to(device),
        PosteriorLoss(stackflow=model.stackflow, 
                      hooffset=model.flow_loss.hooffset, 
                      human_features=predictions['human_features'].detach(), 
                      hoi_features=predictions['hoi_features'].detach(), 
                      object_labels=batch['object_labels']).to(device)
    ]
    iterations = cfg.eval.optim_iters
    steps_per_iter = cfg.eval.optim_steps

    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        for i in loop:
            optimizer.zero_grad()
            hoi_dict = hoi_instance.forward()
            losses = {}
            for f in loss_functions:
                losses.update(f(hoi_dict))
            loss_list = [loss_weights[k](v.mean(), it) for k, v in losses.items()]
            total_loss = torch.stack(loss_list).sum()

            total_loss.backward()
            optimizer.step()

            l_str = 'Optim. Step {}: Iter: {}'.format(it, i)
            for k, v in losses.items():
                l_str += ', {}: {:0.4f}'.format(k, v.mean().detach().item())
                loop.set_description(l_str)
                
    hoi_dict = hoi_instance.forward()
    predictions['pred_betas'] = hoi_dict['smpl_betas']
    predictions['pred_smpl_body_pose'] = hoi_dict['smpl_body_rotmat']
    predictions['pred_obj_rel_T'] = hoi_dict['obj_rel_trans']
    predictions['pred_obj_rel_R'] = hoi_dict['obj_rel_rotmat']
    predictions['hoi_trans'] = hoi_dict['hoi_trans']
    predictions['hoi_rotmat'] = hoi_dict['hoi_rotmat']

    return predictions
