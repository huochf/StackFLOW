import torch
import torch.nn as nn

from nflows.flows import ConditionalGlow


class StackFlow(nn.Module):

    def __init__(self, cfg):
        super(StackFlow, self).__init__()
        self.cfg = cfg
        self.smpl_in_dim = (21 + 1) * 6
        self.smpl_flow = Flow(in_dim=self.smpl_in_dim,
                              hidden_dim=cfg.model.smplflow.hidden_dim,
                              num_layers=cfg.model.smplflow.num_layers,
                              num_blocks_per_layer=cfg.model.smplflow.num_blocks_per_layer,
                              condition_dim=cfg.model.visual_feature_dim)
        self.offset_flow = Flow(in_dim=cfg.model.offset.latent_dim,
                                hidden_dim=cfg.model.offsetflow.hidden_dim,
                                num_layers=cfg.model.offsetflow.num_layers,
                                num_blocks_per_layer=cfg.model.offsetflow.num_blocks_per_layer,
                                condition_dim=cfg.model.visual_feature_dim)

        if cfg.model.offsetflow.object_embedding:
            self.object_embedding = nn.Embedding(cfg.dataset.num_object, cfg.model.visual_feature_dim)
        self.smpl_pose_embedding = nn.Linear(self.smpl_in_dim, cfg.model.visual_feature_dim)


    def forward(self, human_features, hoi_features, object_labels=None, theta_z=None, gamma_z=None):
        b = human_features.shape[0]
        if theta_z is None:
            theta_z = torch.zeros(b, 1, self.smpl_in_dim, dtype=human_features.dtype, device=human_features.device)
        else:
            theta_z = theta_z.unsqueeze(1)
        theta_samples, theta_log_prob, theta_z = self.smpl_flow.sample_and_log_prob(human_features, z=theta_z)

        if gamma_z is None:
            gamma_z = torch.zeros(b, 1, self.cfg.model.offset.latent_dim, dtype=hoi_features.dtype, device=hoi_features.device)
        else:
            gamma_z = gamma_z.unsqueeze(1)
            
        offset_condition = hoi_features
        offset_condition = offset_condition + self.smpl_pose_embedding(theta_samples.squeeze(1))
        if self.cfg.model.offsetflow.object_embedding:
            offset_condition = offset_condition + self.object_embedding(object_labels)
        gamma_samples, gamma_log_prob, gamma_z = self.offset_flow.sample_and_log_prob(offset_condition, z=gamma_z)

        return theta_samples.squeeze(1), theta_log_prob.squeeze(1), theta_z.squeeze(1), gamma_samples.squeeze(1), gamma_log_prob.squeeze(1), gamma_z.squeeze(1)


    def sample(self, num_samples, human_features, hoi_features, object_labels=None):
        b = human_features.shape[0]
        theta_samples, theta_log_prob, theta_z = self.smpl_flow.sample_and_log_prob(human_features, num_samples=num_samples)

        offset_condition = hoi_features
        offset_condition = offset_condition[:, None] + self.smpl_pose_embedding(theta_samples)
        if self.cfg.model.offsetflow.object_embedding:
            offset_condition = offset_condition + self.object_embedding(object_labels)[:, None]
        offset_condition = offset_condition.reshape(b * num_samples, -1)
        gamma_samples, gamma_log_prob, gamma_z = self.offset_flow.sample_and_log_prob(offset_condition, num_samples=1)
        gamma_samples = gamma_samples.reshape(b, num_samples, -1)
        gamma_log_prob = gamma_log_prob.reshape(b, num_samples, )
        gamma_z = gamma_z.reshape(b, num_samples, -1)

        return theta_samples, theta_log_prob, theta_z, gamma_samples, gamma_log_prob, gamma_z


    def log_prob(self, theta, gamma, human_features, hoi_features, object_labels=None):
        # theta: [b, dim], gamma: [b, dim], visual_features: [b, dim], object_labels: [b, ]
        theta_log_prob, theta_z = self.smpl_flow.log_prob(theta.unsqueeze(1), human_features)
        theta_log_prob = theta_log_prob.squeeze(1)
        theta_z = theta_z.squeeze(1)

        offset_condition = hoi_features
        offset_condition = offset_condition + self.smpl_pose_embedding(theta)
        if self.cfg.model.offsetflow.object_embedding:
            offset_condition = offset_condition + self.object_embedding(object_labels)
        gamma_log_prob, gamma_z = self.offset_flow.log_prob(gamma.unsqueeze(1), offset_condition)
        gamma_log_prob = gamma_log_prob.squeeze(1) # (b, )
        gamma_z = gamma_z.squeeze(1) # (b, dim)

        return theta_log_prob, theta_z, gamma_log_prob, gamma_z


class Flow(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, num_blocks_per_layer, condition_dim):
        super(Flow, self).__init__()
        self.flow = ConditionalGlow(features=in_dim,
                                    hidden_features=hidden_dim,
                                    num_layers=num_layers,
                                    num_blocks_per_layer=num_blocks_per_layer,
                                    context_features=condition_dim)


    def sample_and_log_prob(self, c, num_samples=None, z=None):
        # c: [b, dim]
        b = c.shape[0]

        if z is None:
            samples, log_prob, z = self.flow.sample_and_log_prob(num_samples, context=c)
            samples = samples.reshape(b, num_samples, -1)
            log_prob = log_prob.reshape(b, num_samples)
            z = z.reshape(b, num_samples, -1)
        else:
            num_samples = z.shape[1]
            samples, log_prob, z = self.flow.sample_and_log_prob(num_samples, context=c, noise=z)
            samples = samples.reshape(b, num_samples, -1)
            log_prob = log_prob.reshape(b, num_samples)
            z = z.reshape(b, num_samples, -1)
        return samples, log_prob, z


    def log_prob(self, samples, c):
        # samples: [b, n, dim], c: [b, dim]
        b = c.shape[0]
        num_samples = samples.shape[1]
        context = c.unsqueeze(1).repeat(1, num_samples, 1)
        log_prob, z = self.flow.log_prob(samples.reshape(b * num_samples, -1), context=context.reshape(b * num_samples, -1))
        log_prob = log_prob.reshape(b, num_samples)
        z = z.reshape(b, num_samples, -1)
        return log_prob, z
