import torch
import torch.nn as nn


class FCHeader(nn.Module):

    def __init__(self, cfg):
        super(FCHeader, self).__init__()
        self.layers = nn.Sequential(nn.Linear(cfg.model.visual_feature_dim, cfg.model.cam_head_dim),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(cfg.model.cam_head_dim, 13))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        init_cam = cfg.dataset.init_cam_translation
        self.register_buffer('init_cam', torch.tensor(init_cam, dtype=torch.float32).reshape(1, 3))


    def forward(self, c):
        # c: [b, dim]
        offset = self.layers(c)
        pred_betas = offset[:, :10]
        pred_cam = self.init_cam + offset[:, 10:]

        return pred_betas, pred_cam
