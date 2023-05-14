import os
import sys
import json
from typing import Dict
from easydict import EasyDict
import yaml
import argparse
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from stackflow.datasets.behave_hoi_img_dataset import BEHAVEImageDataset
from stackflow.datasets.intercap_hoi_img_dataset import IntercapImageDataset
from stackflow.models.model import Model


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, Dict):
            batch[k] = to_device(v, device)
    return batch


def get_dataloader(cfg):
    if cfg.dataset.name == 'behave':
        train_dataset = BEHAVEImageDataset(cfg, is_train=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, drop_last=True)
        test_dataset = BEHAVEImageDataset(cfg ,is_train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, drop_last=True)
    else:
        train_dataset = IntercapImageDataset(cfg, is_train=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, drop_last=True)
        test_dataset = IntercapImageDataset(cfg ,is_train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader


def train(cfg):
    device = torch.device('cuda')

    model = Model(cfg)
    model.to(device)

    begin_epoch = 0
    if cfg.train.resume and os.path.exists(cfg.train.checkpoint):
        begin_epoch = model.load_checkpoint(cfg.train.checkpoint)


    train_dataloader, test_dataloader = get_dataloader(cfg)

    if not os.path.exists(cfg.train.checkpoint_out_dir):
        os.makedirs(cfg.train.checkpoint_out_dir)

    for epoch in range(begin_epoch, cfg.train.max_epoch):
        if epoch >= 20:
            model.loss_weights['loss_trans'] = 1.
        if cfg.train.drop_at == epoch:
            for param_group in model.optimizer.param_groups:
                param_group['lr'] *= 0.1
        model.train()
        for idx, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            output = model.train_step(batch)

            if idx % cfg.train.log_interval == 0:
                loss_str = '[{}, {}], '.format(epoch, idx)
                for k, v in output['losses'].items():
                    loss_str += '{}: {:.4f}, '.format(k, v.item())
                loss_str += '{}: {:.5f}, '.format('lr', model.optimizer.state_dict()['param_groups'][0]['lr'])
                print(loss_str)
                sys.stdout.flush()

        if epoch % cfg.train.eval_interval == 0:
            model.eval()
            evalulation_losses = []
            for idx, batch in enumerate(test_dataloader):
                batch = to_device(batch, device)
                output = model.validation_step(batch)
                evalulation_losses.append(output['losses'])

                if idx % cfg.train.log_interval == 0:
                    loss_str = 'EVAL: [{}, {}], '.format(epoch, idx)
                    for k, v in output['losses'].items():
                        loss_str += '{}: {:.4f}, '.format(k, v.item())
                    loss_str += '{}: {:.5f}, '.format('lr', model.optimizer.state_dict()['param_groups'][0]['lr'])
                    print(loss_str)
                    sys.stdout.flush()
            avg_eval_losses = {}
            avg_eval_losses['epoch'] = epoch
            for losses in evalulation_losses:
                for k, v in losses.items():
                    if k not in avg_eval_losses:
                        avg_eval_losses[k] = 0
                    avg_eval_losses[k] += v.item() / len(test_dataloader)
            with open(os.path.join(cfg.train.checkpoint_out_dir, '{}_eval_logs.json'.format(cfg.train.exp)), 'a') as f:
                f.write(json.dumps(avg_eval_losses) + "\n")

        model.save_checkpoint(epoch, cfg.train.checkpoint_out_dir, '{}.pth'.format(cfg.train.exp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/model_behave.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    train(cfg)
