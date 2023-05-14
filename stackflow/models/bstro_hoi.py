# adapted from https://github.com/paulchhuang/bstro
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from easydict import EasyDict

from stackflow.models.hrnet import get_cls_net
from stackflow.models.hrnet_config import _C as hrnet_config
from stackflow.models.hrnet_config import update_config as hrnet_update_config

class METRO_Encoder(nn.Module):

    def __init__(self, config):
        super(METRO_Encoder, self).__init__()
        self.config = config

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.img_embedding = nn.Linear(config.img_feature_dim, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        encoder_layer = TransformerEncoderLayer(d_model=config.hidden_size, 
                                                nhead=config.num_attention_heads, 
                                                dim_feedforward=config.intermediate_size, 
                                                batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers, )


    def forward(self, img_feats):
        batch_size = len(img_feats)
        seq_lengths = len(img_feats[0])

        position_ids = torch.arange(seq_lengths, dtype=torch.long, device=img_feats.device).unsqueeze(0).repeat(batch_size, 1)
        position_embeddings = self.position_embeddings(position_ids)

        img_embedding_output = self.img_embedding(img_feats)
        embeddings = position_embeddings + img_embedding_output
        embeddings = self.dropout(embeddings)
        encoder_outputs = self.encoder(embeddings)

        return encoder_outputs


class METRO(nn.Module):

    def __init__(self, config):
        super(METRO, self).__init__()
        self.config = config
        self.bert = METRO_Encoder(config)
        self.cls_head = nn.Linear(config.hidden_size, config.output_feature_dim)
        self.residual = nn.Linear(config.img_feature_dim, config.output_feature_dim)


    def forward(self, img_feats):
        predictions = self.bert(img_feats)
        pred_score = self.cls_head(predictions)
        res_img_feats = self.residual(img_feats)
        pred_score = pred_score + res_img_feats

        return pred_score


class BSTRO_HOI(nn.Module):

    def __init__(self, config, backbone, person_trans_encoder, object_trans_encoder):
        super(BSTRO_HOI, self).__init__()
        self.config = config
        self.backbone = backbone
        self.person_trans_encoder = person_trans_encoder
        self.object_trans_encoder = object_trans_encoder
        self.person_upsampling = nn.Linear(431, 1723)
        self.object_upsampling = nn.Linear(128, 256)
        if config.dataset.name == 'behave':
            self.object_upsampling2 = nn.Linear(256, 1700)
            self.object_embedding = nn.Embedding(20, 2051)
            self.person_upsampling2 = nn.Linear(1723, 6890)
        else:
            self.object_upsampling2 = nn.Linear(256, 2500)
            self.object_embedding = nn.Embedding(10, 2051)
            self.person_upsampling2 = nn.Linear(1723, 10475)
        self.person_conv_learn_tokens = nn.Conv1d(64, 431, 1)
        self.object_conv_learn_tokens = nn.Conv1d(64, 128, 1)

        self.criterion_contact = nn.BCELoss()
        self.loss_weights = {
            'smpl_w_sub2': 0.33,
            'smpl_w_sub1': 0.33,
            'smpl_w_full': 0.33,
            'object_w_sub2': 0.33,
            'object_w_sub1': 0.33,
            'object_w_full': 0.33,
        }


    def forward(self, batch):
        images = batch['image']
        batch_size = images.shape[0]
        image_feat = self.backbone(images)
        image_feat_newview = image_feat.view(batch_size, 2048, -1)
        image_feat_newview = image_feat_newview.transpose(1, 2)

        img_tokens = self.person_conv_learn_tokens(image_feat_newview)
        ref_human_vertices = batch['smpl_ref_vertices']
        features = torch.cat([ref_human_vertices, img_tokens], dim=2)
        features = self.person_trans_encoder(features)

        smpl_pred_vertices_sub2 = features
        temp_transpose = smpl_pred_vertices_sub2.transpose(1, 2)
        smpl_pred_vertices_sub = self.person_upsampling(temp_transpose)
        smpl_pred_vertices_full = self.person_upsampling2(smpl_pred_vertices_sub)
        smpl_pred_vertices_sub = smpl_pred_vertices_sub.transpose(1, 2)
        smpl_pred_vertices_full = smpl_pred_vertices_full.transpose(1, 2)

        img_tokens = self.object_conv_learn_tokens(image_feat_newview)
        ref_object_vertices = batch['object_ref_vertices']
        features = torch.cat([ref_object_vertices, img_tokens], dim=2)
        object_label = batch['object_label']
        features = features + self.object_embedding(object_label).unsqueeze(1)
        features = self.object_trans_encoder(features)

        object_pred_vertices_sub2 = features
        temp_transpose = object_pred_vertices_sub2.transpose(1, 2)
        object_pred_vertices_sub = self.object_upsampling(temp_transpose)
        object_pred_vertices_full = self.object_upsampling2(object_pred_vertices_sub)
        object_pred_vertices_sub = object_pred_vertices_sub.transpose(1, 2)
        object_pred_vertices_full = object_pred_vertices_full.transpose(1, 2)

        return (torch.sigmoid(smpl_pred_vertices_sub2), torch.sigmoid(smpl_pred_vertices_sub), torch.sigmoid(smpl_pred_vertices_full), 
                torch.sigmoid(object_pred_vertices_sub2), torch.sigmoid(object_pred_vertices_sub), torch.sigmoid(object_pred_vertices_full), )


    def train_step(self, batch):
        smpl_sub2, smpl_sub, smpl_full, object_sub2, object_sub, object_full = self.forward(batch)
        loss_smpl_sub2 = self.criterion_contact(smpl_sub2, batch['person_contact_map_l2'])
        loss_smpl_sub = self.criterion_contact(smpl_sub, batch['person_contact_map_l1'])
        loss_smpl_full = self.criterion_contact(smpl_full, batch['person_contact_map'])
        loss_object_sub2 = self.criterion_contact(object_sub2, batch['object_contact_map_l2'])
        loss_object_sub = self.criterion_contact(object_sub, batch['object_contact_map_l1'])
        loss_object_full = self.criterion_contact(object_full * batch['object_vertices_mask'].unsqueeze(2), batch['object_contact_map'])

        loss = self.loss_weights['smpl_w_sub2'] * loss_smpl_sub2 +\
               self.loss_weights['smpl_w_sub1'] * loss_smpl_sub +\
               self.loss_weights['smpl_w_full'] * loss_smpl_full +\
               self.loss_weights['object_w_sub2'] * loss_object_sub2 +\
               self.loss_weights['object_w_sub1'] * loss_object_sub +\
               self.loss_weights['object_w_full'] * loss_object_full
        losses = dict(
            loss = loss,
            loss_smpl_sub2=loss_smpl_sub2,
            loss_smpl_sub=loss_smpl_sub,
            loss_smpl_full=loss_smpl_full,
            loss_object_sub2=loss_object_sub2,
            loss_object_sub=loss_object_sub,
            loss_object_full=loss_object_full,
        )
        return loss, losses


    def validation_step(self, batch):
        with torch.no_grad():
            smpl_sub2, smpl_sub, smpl_full, object_sub2, object_sub, object_full = self.forward(batch)
            loss_smpl_sub2 = self.criterion_contact(smpl_sub2, batch['person_contact_map_l2'])
            loss_smpl_sub = self.criterion_contact(smpl_sub, batch['person_contact_map_l1'])
            loss_smpl_full = self.criterion_contact(smpl_full, batch['person_contact_map'])
            loss_object_sub2 = self.criterion_contact(object_sub2, batch['object_contact_map_l2'])
            loss_object_sub = self.criterion_contact(object_sub, batch['object_contact_map_l1'])
            loss_object_full = self.criterion_contact(object_full, batch['object_contact_map'])

        loss = self.loss_weights['smpl_w_sub2'] * loss_smpl_sub2 +\
               self.loss_weights['smpl_w_sub1'] * loss_smpl_sub +\
               self.loss_weights['smpl_w_full'] * loss_smpl_full +\
               self.loss_weights['object_w_sub2'] * loss_object_sub2 +\
               self.loss_weights['object_w_sub1'] * loss_object_sub +\
               self.loss_weights['object_w_full'] * loss_object_full
        losses = dict(
            loss = loss,
            loss_smpl_sub2=loss_smpl_sub2,
            loss_smpl_sub=loss_smpl_sub,
            loss_smpl_full=loss_smpl_full,
            loss_object_sub2=loss_object_sub2,
            loss_object_sub=loss_object_sub,
            loss_object_full=loss_object_full,
        )
        return losses


def get_default_config():

    config = {
        'max_position_embeddings': 512,
        'hidden_dropout_prob': 0.1,
        'num_attention_heads': 8,
        'num_hidden_layers': 4,
        'output_feature_dim': 2,
    }
    return config


def build_BSTRO_HOI(config):
    hrnet_yaml = 'data/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
    hrnet_checkpoint = 'data/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
    hrnet_update_config(hrnet_config, hrnet_yaml)
    backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)

    input_feat_dims = [2051, 512, 128]
    hidden_feat_dims = [1024, 256, 128]
    output_feat_dims = input_feat_dims[1:] + [1]

    object_trans_encoder = []
    encoder_config = get_default_config()
    for i in range(len(output_feat_dims)):
        encoder_config['img_feature_dim'] = input_feat_dims[i]
        encoder_config['output_feature_dim'] = output_feat_dims[i]
        encoder_config['hidden_size'] = hidden_feat_dims[i]
        encoder_config['intermediate_size'] = hidden_feat_dims[i] * 4

        encoder_config = EasyDict(encoder_config)
        model = METRO(encoder_config)
        object_trans_encoder.append(model)
    object_trans_encoder = nn.Sequential(*object_trans_encoder)

    person_trans_encoder = []
    encoder_config = get_default_config()
    for i in range(len(output_feat_dims)):
        encoder_config['img_feature_dim'] = input_feat_dims[i]
        encoder_config['output_feature_dim'] = output_feat_dims[i]
        encoder_config['hidden_size'] = hidden_feat_dims[i]
        encoder_config['intermediate_size'] = hidden_feat_dims[i] * 4

        encoder_config = EasyDict(encoder_config)
        model = METRO(encoder_config)
        person_trans_encoder.append(model)
    person_trans_encoder = nn.Sequential(*person_trans_encoder)

    total_params = sum(p.numel() for p in person_trans_encoder.parameters())
    print('total person transformer encoder: {} params'.format(total_params))

    bstro = BSTRO_HOI(config, backbone, person_trans_encoder, object_trans_encoder)

    return bstro
