import torch
import torch.nn as nn
import torchvision.models as models


def build_backbone(cfg):
    if cfg.model.backbone == 'resnet':
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        modules.append(nn.Conv2d(resnet.fc.in_features, cfg.model.visual_feature_dim, kernel_size=1))
        backbone = nn.Sequential(*modules)
    elif cfg.model.backbone == 'ViT':
        backbone = ViT(image_size=cfg.dataset.img_size, 
                       patch_size=cfg.model.patch_size,
                       dim=cfg.model.dim,
                       condition_dim=cfg.model.visual_feature_dim,
                       depth=cfg.model.depth,
                       heads=cfg.model.heads,
                       mlp_dim=cfg.model.mlp_dim,
                       dropout=cfg.model.dropout,
                       emb_dropout=cfg.model.emb_dropout)
    else:
        raise NotImplementedError

    return backbone



########################### copy from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py #####################################

# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange


# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)


# class PreNorm(nn.Module):

#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn


#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)


# class FeedForward(nn.Module):

#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )


#     def forward(self, x):
#         return self.net(x)


# class Attention(nn.Module):

#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not(heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5
        
#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()


#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)


# class Transformer(nn.Module):

#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
#             ]))


#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x


# class ViT(nn.Module):

#     def __init__(self, *, image_size, patch_size, dim, condition_dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)

#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim),
#             nn.LayerNorm(dim),
#         )

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 3, dim))
#         self.global_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.human_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.hoi_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.global_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, condition_dim)
#         )
#         self.human_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, condition_dim)
#         )
#         self.hoi_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, condition_dim)
#         )


#     def forward(self, img):
#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape

#         global_tokens = repeat(self.global_token, '1 1 d -> b 1 d', b=b)
#         human_tokens = repeat(self.human_token, '1 1 d -> b 1 d', b=b)
#         hoi_tokens = repeat(self.hoi_token, '1 1 d -> b 1 d', b=b)
#         x = torch.cat((global_tokens, human_tokens, hoi_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 3)]
#         x = self.dropout(x)

#         x = self.transformer(x)

#         x_global = self.global_head(x[:, 0])
#         x_human = self.human_head(x[:, 1])
#         x_hoi = self.hoi_head(x[:, 2])
#         return x_global, x_human, x_hoi


#################################################################### end of copy #################################################################
