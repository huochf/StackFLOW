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
import pickle

from stackflow.datasets.behave_hoi_contact_dataset import BEHAVEContactDataset
from stackflow.datasets.intercap_hoi_contact_dataset import IntercapContactDataset
from stackflow.models.bstro_hoi import build_BSTRO_HOI


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, Dict):
            batch[k] = to_device(v, device)
    return batch


def get_dataloader(cfg):
    if cfg.dataset.name == 'behave':
        train_dataset = BEHAVEContactDataset(cfg, is_train=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, drop_last=True)
        test_dataset = BEHAVEContactDataset(cfg ,is_train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, drop_last=False)
    else:
        train_dataset = IntercapContactDataset(cfg, is_train=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, drop_last=True)
        test_dataset = IntercapContactDataset(cfg ,is_train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, drop_last=False)

    return train_dataloader, test_dataloader


def load_checkpoint(model, optimizer, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
    return state_dict['epoch']


def save_checkpoint(epoch, model, optimizer, dir, name,):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(dir, name))


def train(cfg):
    device = torch.device('cuda')

    model = build_BSTRO_HOI(cfg)
    model.to(device)

    optimizer = torch.optim.Adam(params=list(model.parameters()),
                                           lr=1e-4,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)

    begin_epoch = 0
    if cfg.train.resume and os.path.exists(cfg.train.checkpoint):
        begin_epoch = load_checkpoint(model, optimizer, cfg.train.checkpoint)

    train_dataloader, test_dataloader = get_dataloader(cfg)

    if not os.path.exists(cfg.train.checkpoint_out_dir):
        os.makedirs(cfg.train.checkpoint_out_dir)


    for epoch in range(begin_epoch, cfg.train.max_epoch):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            loss, losses = model.train_step(batch)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            if idx % cfg.train.log_interval == 0:
                loss_str = '[{}, {}], '.format(epoch, idx)
                for k, v in losses.items():
                    loss_str += '{}: {:.4f}, '.format(k, v.item())
                loss_str += '{}: {:.5f}, '.format('lr', optimizer.state_dict()['param_groups'][0]['lr'])
                print(loss_str)
                sys.stdout.flush()

        if epoch % cfg.train.eval_interval == 0:
            model.eval()
            evalulation_losses = []
            for idx, batch in enumerate(test_dataloader):
                batch = to_device(batch, device)
                losses = model.validation_step(batch)
                evalulation_losses.append(losses)

                if idx % cfg.train.log_interval == 0:
                    loss_str = 'EVAL: [{}, {}], '.format(epoch, idx)
                    for k, v in losses.items():
                        loss_str += '{}: {:.4f}, '.format(k, v.item())
                    loss_str += '{}: {:.5f}, '.format('lr', optimizer.state_dict()['param_groups'][0]['lr'])
                    print(loss_str)
                    sys.stdout.flush()
            avg_eval_losses = {}
            for losses in evalulation_losses:
                for k, v in losses.items():
                    if k not in avg_eval_losses:
                        avg_eval_losses[k] = 0
                    avg_eval_losses[k] += v.item() / len(test_dataloader)
            with open(os.path.join(cfg.train.checkpoint_out_dir, '{}_eval_logs.json'.format(cfg.train.exp)), 'a') as f:
                f.write(json.dumps(avg_eval_losses) + "\n")

        save_checkpoint(epoch, model, optimizer, cfg.train.checkpoint_out_dir, '{}.pth'.format(cfg.train.exp))


def inference(cfg):
    device = torch.device('cuda')

    model = build_BSTRO_HOI(cfg)
    model.to(device)
    load_checkpoint(model, None, 'outputs/bstro_hoi/behave_bstro_hoi_no_aug.pth')

    train_dataloader, test_dataloader = get_dataloader(cfg)

    results = {}
    model.eval()
    for idx, batch in enumerate(test_dataloader):
        batch = to_device(batch, device)
        with torch.no_grad():
            smpl_sub2, smpl_sub, smpl_full, object_sub2, object_sub, object_full = model(batch)

        img_ids = batch['img_id']
        for b in range(len(img_ids)):
            img_id = img_ids[b]
            results[img_id] = {
                # 'smpl_sub2': smpl_sub2[b].detach().cpu().numpy(),
                # 'smpl_sub': smpl_sub[b].detach().cpu().numpy(),
                'smpl_full': smpl_full[b].detach().cpu().numpy(),
                # 'object_sub2': object_sub2[b].detach().cpu().numpy(),
                # 'object_sub': object_sub[b].detach().cpu().numpy(),
                'object_full': object_full[b].detach().cpu().numpy(),
                # 'smpl_sub2_gt': batch['person_contact_map_l2'][b].detach().cpu().numpy(),
                # 'smpl_sub_gt': batch['person_contact_map_l1'][b].detach().cpu().numpy(),
                'smpl_full_gt': batch['person_contact_map'][b].detach().cpu().numpy(),
                # 'object_sub2_gt': batch['object_contact_map_l2'][b].detach().cpu().numpy(),
                # 'object_sub_gt': batch['object_contact_map_l1'][b].detach().cpu().numpy(),
                'object_full_gt': batch['object_contact_map'][b].detach().cpu().numpy(),
            }
            print('{} / {}, {} done!'.format(idx, len(test_dataloader), img_id))
        # if idx > 10:
        #     break

    with open('outputs/bstro_hoi/behave_bstro_hoi_no_aug_outputs.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/bstro_hoi_behave.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    train(cfg)
    # inference(cfg)
