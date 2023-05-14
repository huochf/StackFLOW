import torch
import torch.nn as nn

from nflows.flows import ConditionalGlow
from pytorch3d.transforms import rotation_6d_to_matrix


class SMPLHead(nn.Module):

    def __init__(self, cfg):
        super(SMPLHead, self).__init__()
        self.layers = nn.Sequential(nn.Linear(cfg.model.smpl_header.in_dim, cfg.model.smpl_header.hidden_dim),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(cfg.model.smpl_header.hidden_dim, 13))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        if cfg.dataset.name == 'behave':
            init_cam = cfg.dataset.behave.init_cam_translation
        else:
            init_cam = cfg.dataset.intercap.init_cam_translation
        self.register_buffer('init_cam', torch.tensor(init_cam, dtype=torch.float32).reshape(1, 3))


    def forward(self, features):
        offset = self.layers(features)
        pred_betas = offset[:, :10]
        pred_cam = self.init_cam + offset[:, 10:]

        return pred_betas, pred_cam


class HOIFlow(nn.Module):

    def __init__(self, cfg):
        super(HOIFlow, self).__init__()
        self.smpl_pose = (cfg.model.hoiflow.num_pose + 1) * 6 # 6 * (21 + 1)
        self.hoi_dim = cfg.model.hoi_decoder.pca_dim
        self.num_pose = cfg.model.hoiflow.num_pose
        self.smpl_flow = ConditionalGlow(features=self.smpl_pose, 
                                         hidden_features=cfg.model.hoiflow.hidden_dim, 
                                         num_layers=cfg.model.hoiflow.num_layers,
                                         num_blocks_per_layer=cfg.model.hoiflow.num_blocks_per_layer,
                                         context_features=cfg.model.hoiflow.context_features)
        self.hoi_flow = ConditionalGlow(features=cfg.model.hoi_decoder.pca_dim,
                                        hidden_features=cfg.model.hoiflow.hidden_dim,
                                        num_layers=cfg.model.hoiflow.num_layers,
                                        num_blocks_per_layer=cfg.model.hoiflow.num_blocks_per_layer,
                                        context_features=cfg.model.hoiflow.context_features)
        self.pose_embedding = nn.Linear(self.smpl_pose, cfg.model.hoiflow.context_features)
        self.beta_embedding = nn.Linear(10, cfg.model.hoiflow.context_features)
        self.object_embedding = nn.Embedding(cfg.data.object_num, cfg.model.hoiflow.context_features)
        self.smpl_header = SMPLHead(cfg)


    def forward(self, features, object_labels, num_samples=None, z=(None, None)):
        batch_size = features.shape[0]

        assert not (num_samples is None and z is None), 'you should provide either z or num_samples'

        pred_betas, pred_cam = self.smpl_header(features)

        smpl_z, hoi_z = z
        if num_samples is None:
            num_samples = smpl_z.shape[1]

        pose_samples, pose_log_prob, pose_z = self.smpl_flow.sample_and_log_prob(num_samples, context=features, noise=smpl_z)
        pose_samples = pose_samples.reshape(batch_size, num_samples, -1)
        pose_log_prob = pose_log_prob.reshape(batch_size, num_samples)
        pose_z = pose_z.reshape(batch_size, num_samples, -1)

        pose_features = self.pose_embedding(pose_samples).reshape(batch_size * num_samples, -1)
        beta_features = self.beta_embedding(pred_betas).unsqueeze(1).repeat(1, num_samples, 1).reshape(batch_size * num_samples, -1)
        object_features = self.object_embedding(object_labels).unsqueeze(1).repeat(1, num_samples, 1).reshape(batch_size * num_samples, -1)
        hoi_conditions = features.unsqueeze(1).repeat(1, num_samples, 1).reshape(batch_size * num_samples, -1)
        hoi_conditions += pose_features
        hoi_conditions += beta_features
        hoi_conditions += object_features
        hoi_samples, hoi_log_prob, hoi_z = self.hoi_flow.sample_and_log_prob(num_samples, context=hoi_conditions, noise=hoi_z)
        hoi_samples = hoi_samples.reshape(batch_size, num_samples, num_samples, -1)
        hoi_log_prob = hoi_log_prob.reshape(batch_size, num_samples, num_samples)
        hoi_z = hoi_z.reshape(batch_size, num_samples, num_samples, -1)

        pred_pose_6d = pose_samples.reshape(batch_size, num_samples, self.num_pose + 1, 6)
        pred_pose_rotmat = rotation_6d_to_matrix(pred_pose_6d.clone())

        smpl_samples = {
            'pose_6d': pred_pose_6d, # (b, n, 22, 6)
            'pose_rotmat': pred_pose_rotmat, # (b, n, 22, 3, 3)
            'betas': pred_betas, # (b, 10)
            'log_prob': pose_log_prob, # (b, n)
            'z': pose_z, # (b, n, c)
        }

        hoi_samples = {
            'hoi_lattent_codes': hoi_samples, # (b, n, n, 32)
            'log_prob': hoi_log_prob, # (b, n, n)
            'z': hoi_z, # (b, n, n, c)
        }

        return smpl_samples, hoi_samples, pred_cam


    def log_prob(self, smpl_pose, smpl_beta, hoi_lattent_codes, features, object_labels):
        # smpl_pose: (b, 22 * 6)
        # smpl_beta: (b, 10)
        # hoi_lattent_codes: (b, 32)
        batch_size = features.shape[0]

        smpl_log_prob, smpl_z = self.smpl_flow.log_prob(smpl_pose, features)
        smpl_log_prob = smpl_log_prob.reshape(batch_size, )
        smpl_z = smpl_z.reshape(batch_size, -1)

        pose_embedding = self.pose_embedding(smpl_pose)
        beta_embedding = self.beta_embedding(smpl_beta)
        object_features = self.object_embedding(object_labels)
        hoi_conditions = features + pose_embedding + beta_embedding + object_features
        # hoi_conditions = beta_embedding + features + object_features
        hoi_log_prob, hoi_z = self.hoi_flow.log_prob(hoi_lattent_codes, hoi_conditions)
        hoi_log_prob = hoi_log_prob.reshape(batch_size)
        hoi_z = hoi_z.reshape(batch_size, -1)

        return smpl_log_prob, smpl_z, hoi_log_prob, hoi_z
