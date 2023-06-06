import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import argparse
import numpy as np
from tqdm import tqdm

from stackflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from stackflow.datasets.utils import save_pickle


def main(args):
    # merge annotations
    seq_names = [('Date02_Sub02_monitor_move', 'Date02_Sub02_monitor_move2'), 
                 ('Date02_Sub02_toolbox', 'Date02_Sub02_toolbox_part2'), 
                 ('Date03_Sub04_boxtiny', 'Date03_Sub04_boxtiny_part2'),
                 ('Date03_Sub04_yogaball_play', 'Date03_Sub04_yogaball_play2'),
                 ('Date03_Sub05_chairwood', 'Date03_Sub05_chairwood_part2'),
                 ('Date04_Sub05_monitor', 'Date04_Sub05_monitor_part2')]

    for pair in seq_names:
        name1, name2 = pair
        object_fit1 = np.load(os.path.join(args.root_dir, 'behave-30fps-params-v1', name1, 'object_fit_all.npz'))
        smpl_fit1 = np.load(os.path.join(args.root_dir, 'behave-30fps-params-v1', name1, 'smpl_fit_all.npz'))
        object_fit2 = np.load(os.path.join(args.root_dir, 'behave-30fps-params-v1', name2, 'object_fit_all.npz'))
        smpl_fit2 = np.load(os.path.join(args.root_dir, 'behave-30fps-params-v1', name2, 'smpl_fit_all.npz'))

        object_fit = {}
        object_fit['angles'] = np.concatenate([object_fit1['angles'], object_fit2['angles']], axis=0)
        object_fit['trans'] = np.concatenate([object_fit1['trans'], object_fit2['trans']], axis=0)
        object_fit['frame_times'] = np.concatenate([object_fit1['frame_times'], object_fit2['frame_times']], axis=0)

        smpl_fit = {}
        smpl_fit['poses'] = np.concatenate([smpl_fit1['poses'], smpl_fit2['poses']], axis=0)
        smpl_fit['betas'] = np.concatenate([smpl_fit1['betas'], smpl_fit2['betas']], axis=0)
        smpl_fit['trans'] = np.concatenate([smpl_fit1['trans'], smpl_fit2['trans']], axis=0)
        smpl_fit['frame_times'] = np.concatenate([smpl_fit1['frame_times'], smpl_fit2['frame_times']], axis=0)

        np.savez(os.path.join(args.root_dir, 'behave-30fps-params-v1', name1, 'object_fit_all.npz'), 
            angles=object_fit['angles'], trans=object_fit['trans'], frame_times=object_fit['frame_times'])
        np.savez(os.path.join(args.root_dir, 'behave-30fps-params-v1', name1, 'smpl_fit_all.npz'), 
            poses=smpl_fit['poses'], betas=smpl_fit['betas'], trans=smpl_fit['trans'], frame_times=smpl_fit['frame_times'])

    valid_image_ids = []
    dataset_metadata = BEHAVEExtendMetaData(args.root_dir)
    all_sequences = os.listdir(os.path.join(args.root_dir, 'behave-30fps-params-v1'))
    for seq_name in tqdm(all_sequences):
        if seq_name[-1] == '2':
            continue
        smpl_fit = np.load(os.path.join(args.root_dir, 'behave-30fps-params-v1', seq_name, 'smpl_fit_all.npz'))
        object_fit = np.load(os.path.join(args.root_dir, 'behave-30fps-params-v1', seq_name, 'object_fit_all.npz'))
        if seq_name == 'Date02_Sub02_backpack_twohand':
            # len(smpl_fit['frame_times']) == 1419, len(object_fit['frame_times']) == 1261
            num_frames = len(object_fit['frame_times'])
            temp_smpl_fit = {}
            temp_smpl_fit['poses'] = smpl_fit['poses'][:num_frames]
            temp_smpl_fit['betas'] = smpl_fit['betas'][:num_frames]
            temp_smpl_fit['trans'] = smpl_fit['trans'][:num_frames]
            temp_smpl_fit['frame_times'] = smpl_fit['frame_times'][:num_frames]
            smpl_fit = temp_smpl_fit
        elif seq_name == 'Date06_Sub07_backpack_back':
            # len(smpl_fit['frame_times']) == 1441, len(object_fit['frame_times']) == 1411
            temp_smpl_fit = {}
            temp_smpl_fit['poses'] = smpl_fit['poses'][30:]
            temp_smpl_fit['betas'] = smpl_fit['betas'][30:]
            temp_smpl_fit['trans'] = smpl_fit['trans'][30:]
            temp_smpl_fit['frame_times'] = smpl_fit['frame_times'][30:]
            smpl_fit = temp_smpl_fit

        num_frames = len(smpl_fit['frame_times'])
        assert smpl_fit['poses'].shape[0] == num_frames
        assert smpl_fit['betas'].shape[0] == num_frames
        assert smpl_fit['trans'].shape[0] == num_frames
        assert len(object_fit['frame_times']) == num_frames, seq_name
        assert object_fit['angles'].shape[0] == num_frames, seq_name
        assert object_fit['trans'].shape[0] == num_frames
        assert len(object_fit['frame_times']) == num_frames
        assert (smpl_fit['frame_times'] == object_fit['frame_times']).all()

        day_id, sub_id, obj_name, inter_type = dataset_metadata.parse_seq_info(seq_name)

        for frame_name in smpl_fit['frame_times']:
            frame_id = frame_name[2:]
            for cam_id in range(4):
                img_id = '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, str(cam_id)])
                valid_image_ids.append(img_id)
    print('Total {} valid frames'.format(len(valid_image_ids)))
    save_pickle(valid_image_ids, os.path.join(args.root_dir, 'behave_extend_valid_frames.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str, help='Dataset root directory.')
    args = parser.parse_args()

    main(args)
