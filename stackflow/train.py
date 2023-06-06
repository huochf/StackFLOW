import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import argparse
import json

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from stackflow.configs import load_config
from stackflow.datasets.behave_hoi_dataset import BEHAVEDataset
from stackflow.datasets.intercap_hoi_dataset import InterCapDataset
from stackflow.datasets.behave_extend_hoi_dataset import BEHAVEExtendDataset
from stackflow.datasets.utils import load_pickle
from stackflow.models import Model
from stackflow.utils.visualize import visualize_step
from stackflow.utils.utils import to_device, set_seed



def train(cfg):
    device = torch.device('cuda')

    if cfg.dataset.name == 'BEHAVE':
        train_dataset = BEHAVEDataset(cfg, is_train=True)
        test_dataset = BEHAVEDataset(cfg, is_train=False)
    elif cfg.dataset.name == 'InterCap':
        train_dataset = InterCapDataset(cfg, is_train=True)
        test_dataset = InterCapDataset(cfg, is_train=False)
    elif cfg.dataset.name == 'BEHAVE-Extended':
        annotations = load_pickle(cfg.dataset.annotation_file_train)
        annotations_shared = {item['img_id']: item for item in annotations}
        train_dataset = BEHAVEExtendDataset(cfg, annotations_shared, is_train=True)

        annotations = load_pickle(cfg.dataset.annotation_file_test)
        annotations_shared = {item['img_id']: item for item in annotations}
        test_dataset = BEHAVEExtendDataset(cfg, annotations_shared, is_train=False)
    else:
        raise ValueError('Unsupported dataset {}.'.format(cfg.dataset.name))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=cfg.train.batch_size,
                                                   num_workers=cfg.train.num_workers,
                                                   shuffle=True,
                                                   drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size=cfg.train.batch_size,
                                                   num_workers=2,
                                                   shuffle=True,
                                                   drop_last=True)

    model = Model(cfg)
    model.to(device)

    begin_epoch = 0
    if cfg.train.resume and os.path.exists(cfg.train.resume):
        begin_epoch = model.load_checkpoint(cfg.train.resume)

    for epoch in range(begin_epoch, cfg.train.max_epoch):
        if epoch == cfg.train.drop_lr_at:
            for param_group in model.optimizer.param_groups:
                param_group['lr'] *= 0.1
        model.train()
        if epoch > cfg.train.trans_begin_epoch:
            model.loss_weights['loss_trans'] = 0.1
        else:
            model.loss_weights['loss_trans'] = 0.
        for idx, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            loss, all_losses = model.train_step(batch)

            if idx % cfg.train.log_interval == 0:
                loss_str = '[{}, {}], loss: {:.4f}'.format(epoch, idx, loss.item())
                for k, v in all_losses.items():
                    loss_str += ', {}: {:.4f}'.format(k, v.item())
                loss_str += ', {}: {:.5f}'.format('lr', model.optimizer.state_dict()['param_groups'][0]['lr'])
                print(loss_str)
                sys.stdout.flush()
        model.eval()

        eval_losses = {}
        for idx, batch in enumerate(test_dataloader):
            if idx > 100:
                break
            batch = to_device(batch, device)
            loss, all_losses = model.forward_train(batch) # no loss backward !!!

            if idx % 10 == 0:
                loss_str = 'EVAL: [{}, {}], loss: {:.4f}'.format(epoch, idx, loss.item())
                for k, v in all_losses.items():
                    loss_str += ', {}: {:.4f}'.format(k, v.item())
                print(loss_str)
                sys.stdout.flush()

            # if idx % 10 == 0:
            #     pred = model.inference(batch, debug=True)
            #     visualize_step(cfg, test_dataset.dataset_metadata, batch, pred, epoch, idx)
            for k, v in all_losses.items():
                if k not in eval_losses:
                    eval_losses[k] = 0
                
                eval_losses[k] += v.item() / 100

        os.makedirs(cfg.train.output_dir, exist_ok=True)

        eval_losses['epoch'] = epoch
        with open(os.path.join(cfg.train.output_dir, 'logs.json'), 'a') as f:
            f.write(json.dumps(eval_losses) + "\n")
        model.save_checkpoint(epoch, os.path.join(cfg.train.output_dir, 'latest_checkpoint.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', default='stackflow/configs/behave.yaml', type=str)
    parser.add_argument('--dataset_root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str)
    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    cfg.dataset.root_dir = args.dataset_root_dir
    cfg.freeze()
    set_seed(7)
    train(cfg)
