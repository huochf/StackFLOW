import os
from tqdm import tqdm
import trimesh
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from stackflow.datasets.utils import load_json, load_pickle


class BEHAVEExtendMetaData():

    def __init__(self, root_dir, preload_annotations=False, ):
        self.root_dir = root_dir
        self.IMG_HEIGHT = 1536
        self.IMG_WIDTH = 2048

        self.OBJECT_NAME2IDX = {
            'backpack': 0,
            'basketball': 1,
            'boxlarge': 2,
            'boxlong': 3,
            'boxmedium': 4,
            'boxsmall': 5,
            'boxtiny': 6,
            'chairblack': 7,
            'chairwood': 8,
            'keyboard': 9,
            'monitor': 10,
            'plasticcontainer': 11,
            'stool': 12,
            'suitcase': 13,
            'tablesmall': 14,
            'tablesquare': 15,
            'toolbox': 16,
            'trashbin': 17,
            'yogaball': 18,
            'yogamat': 19,
        }
        self.OBJECT_IDX2NAME = {v: k for k, v in self.OBJECT_NAME2IDX.items()}
        self.OBJECT_COORDINATE_NORM = {
            'basketball': [0.12236282829610301, 0.13042416009636515, 0.13102541633815146], 
            'tablesmall': [0.29788815447386074, 0.4007032965176957, 0.27832651484960635], 
            'toolbox': [0.09294969423491206, 0.13222591918103288, 0.1673733647629312], 
            'plasticcontainer': [0.3358690018561631, 0.22639700585163278, 0.25677905980620397], 
            'boxlarge': [0.26936167564988484, 0.30247685136558083, 0.28287613106284965], 
            'trashbin': [0.15079510307636423, 0.1896149735794419, 0.1603298959826267], 
            'stool': [0.23256295110606504, 0.27003098172428946, 0.2205380318634632], 
            'boxsmall': [0.16008218470133753, 0.14842027491577947, 0.22244380626078553], 
            'monitor': [0.13533075340542058, 0.2979130832046061, 0.29667681821373404], 
            'keyboard': [0.10469683236514524, 0.16815164813278008, 0.2406570611341632], 
            'boxmedium': [0.14518500688264757, 0.2078229492491641, 0.25048296294494005], 
            'chairblack': [0.303116101212625, 0.4511368757997035, 0.2987161170926357], 
            'chairwood': [0.32013054983251643, 0.48153881571638113, 0.37033998297393567], 
            'suitcase': [0.16022201445086703, 0.2550602338788379, 0.2613365624202387], 
            'boxlong': [0.39511341702499553, 0.1720738671379548, 0.1971366981998387], 
            'boxtiny': [0.11570125012439958, 0.060232502239181196, 0.1634993526289597], 
            'yogaball': [0.27815387740465014, 0.26961738674524627, 0.3164645608250861], 
            'backpack': [0.2202841718619516, 0.2839561989281594, 0.19267741049215822], 
            'yogamat': [0.524749682746465, 0.2720144866073263, 0.12567161343996003], 
            'tablesquare': [0.4920387357121939, 0.48840298724966774, 0.48018395294091076]
        }
        self.SUBID_GENDER = {
            '1': 'male',
            '2': 'male',
            '3': 'male',
            '4': 'male',
            '5': 'male',
            '6': 'female',
            '7': 'female',
            '8': 'female',
        }

        self.obj_mesh_templates = self.load_object_mesh_templates()
        self.obj_keypoints_dict = self.load_object_keypoints_dict()

        self.dataset_splits = load_json(os.path.join(root_dir, 'split.json'))
        self.cam_intrinsics = self.load_cam_intrinsics()
        self.cam_RT_matrix = self.load_cam_RT_matrix()
        self.annotations = {}
        if preload_annotations:
            all_sequences = list(self.go_through_all_sequences(split='all'))
            for sequence_name in tqdm(all_sequences, desc='loading annotations'):
                try:
                    self.annotations[sequence_name] = self.load_annotations(sequence_name)
                except:
                    continue

        self.object_max_keypoint_num = 16
        self.object_num_keypoints = {
            'backpack': 8,
            'basketball': 1,
            'boxlarge': 8,
            'boxlong': 8,
            'boxmedium': 8,
            'boxsmall': 8,
            'boxtiny': 8,
            'chairblack': 16,
            'chairwood': 10,
            'keyboard': 4,
            'monitor': 8,
            'plasticcontainer': 8,
            'stool': 6,
            'suitcase': 8,
            'tablesmall': 10,
            'tablesquare': 8,
            'toolbox': 8,
            'trashbin': 2,
            'yogaball': 1,
            'yogamat': 2,
        }
        self.object_max_vertices_num = 1700
        self.object_num_vertices = {
            "backpack": 548,
            "basketball": 508,
            "boxlarge": 524,
            "boxlong": 526,
            "boxmedium": 509,
            "boxsmall": 517,
            "boxtiny": 506,
            "chairblack": 1609,
            "chairwood": 1609,
            "keyboard": 502,
            "monitor": 537,
            "plasticcontainer": 562,
            "stool": 532,
            "suitcase": 520,
            "tablesmall": 507,
            "tablesquare": 1046,
            "toolbox": 499,
            "trashbin": 547,
            "yogaball": 534,
            "yogamat": 525,
        }
        self.all_valid_frames = None


    def load_cam_intrinsics(self, ):
        cam_intrinsics_params = []
        for i in range(4):
            params = load_json(os.path.join(self.root_dir, 'calibs', 'intrinsics', str(i), 'calibration.json'))
            cx, cy = params['color']['cx'], params['color']['cy']
            fx, fy = params['color']['fx'], params['color']['fy']
            cam_intrinsics_params.append([cx, cy, fx, fy])
        return cam_intrinsics_params


    def load_cam_RT_matrix(self, ):
        cam_RT = {}
        for day_id in range(7):
            cam_RT[str(day_id + 1)] = []
            for cam_id in range(4):
                params = load_json(os.path.join(self.root_dir, 'calibs', 'Date0{}'.format(day_id + 1), 'config', str(cam_id), 'config.json'))
                cam_RT[str(day_id + 1)].append([np.array(params['rotation']).reshape((3, 3)), np.array(params['translation'])])
        return cam_RT


    def load_object_mesh_templates(self, ):
        templates = {}
        for object_name in self.OBJECT_NAME2IDX.keys():
            object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f1000.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f2000.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f2500.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_closed_f1000.ply'.format(object_name))
            assert os.path.exists(object_mesh)

            object_mesh = trimesh.load(object_mesh, process=False)
            object_vertices = np.array(object_mesh.vertices).astype(np.float32)
            object_faces = np.array(object_mesh.faces).astype(np.int64)
            object_vertices = object_vertices - object_vertices.mean(axis=0)

            templates[object_name] = (object_vertices, object_faces)

        return templates


    def load_object_trimesh(self, ):
        templates = {}
        for object_name in self.OBJECT_NAME2IDX.keys():
            object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f1000.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f2000.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f2500.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_closed_f1000.ply'.format(object_name))
            assert os.path.exists(object_mesh)

            object_mesh = trimesh.load(object_mesh, process=False)
            object_vertices = np.array(object_mesh.vertices).astype(np.float32)
            object_mesh.vertices = object_vertices - object_vertices.mean(axis=0)

            templates[object_name] = object_mesh

        return templates


    def parse_object_name(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        return obj_name


    def load_object_keypoints_dict(self, ):
        keypoints_dict = {}
        for object_name in self.OBJECT_NAME2IDX.keys():
            keypoints_path = os.path.join('./data/datasets/behave_obj_keypoints/{}_keypoints.json'.format(object_name))
            keypoints = load_json(keypoints_path)
            keypoints_dict[object_name] = keypoints
        return keypoints_dict


    def load_object_keypoints(self, obj_name):
        object_vertices, _ = self.obj_mesh_templates[obj_name]
        keypoints_dict = self.obj_keypoints_dict[obj_name]
        keypoints = []
        for k, v in keypoints_dict.items():
            keypoints.append(object_vertices[v].mean(0))
        keypoints = np.array(keypoints)
        return keypoints


    def parse_seq_info(self, sequence_name):
        try:
            day_name, sub_name, obj_name, inter_type = sequence_name.split('_')
        except:
            day_name, sub_name, obj_name = sequence_name.split('_')
            inter_type = 'none'
        day_id, sub_id = day_name[5:], sub_name[4:]

        return day_id, sub_id, obj_name, inter_type


    def go_through_all_frames(self, split='all'):
        if self.all_valid_frames is None:
            self.all_valid_frames = load_pickle(os.path.join(self.root_dir, 'behave_extend_valid_frames.pkl'))
        all_valid_frames = self.all_valid_frames

        if split == 'train':
            sequences = self.dataset_splits['train']
            all_valid_frames = [img_id for img_id in self.all_valid_frames if self.in_train_set(img_id)]
        if split == 'test':
            sequences = self.dataset_splits['train']
            all_valid_frames = [img_id for img_id in self.all_valid_frames if not self.in_train_set(img_id)]

        for img_id in all_valid_frames:
            yield img_id


    def go_through_all_sequences(self, split='all'):
        
        if split == 'test':
            all_sequences = self.dataset_splits['test']
        elif split == 'train':
            all_sequences = self.dataset_splits['train']
        elif split == 'all':
            all_sequences = self.dataset_splits['train'] + self.dataset_splits['test']

        for name in all_sequences:
            yield name


    def go_through_sequence(self, sequence_name):
        if self.all_valid_frames is None:
            self.all_valid_frames = load_pickle(os.path.join(self.root_dir, 'behave_extend_valid_frames.pkl'))

        target_frames = []
        for img_id in self.all_valid_frames:
            day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
            seq = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
            if seq == sequence_name:
                target_frames.append(img_id)
        target_frames = sorted(target_frames)

        for img_id in target_frames:
            yield img_id


    def get_all_image_by_sequence(self, split='all'):
        all_sequences = list(self.go_through_all_sequences(split))
        img_ids_by_seq = {}
        for sequence_name in all_sequences:
            img_ids_by_seq[sequence_name] = {}

            day_id, sub_id, obj_name, inter_type = self.parse_seq_info(sequence_name)
            for cam_id in range(4):
                img_ids = []
                frame_list = sorted(os.listdir(os.path.join(self.root_dir, 'raw_images', sequence_name)))

                for frame_name in frame_list:
                    if frame_name == 'info.json':
                        continue
                    frame_id = frame_name[2:]
                    img_id =  '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, str(cam_id)])

                    img_ids.append(img_id)
                img_ids_by_seq[sequence_name][cam_id] = img_ids
        return img_ids_by_seq


    def get_img_id(self, day_id, sub_id, obj_name, inter_type, frame_id, cam_id):
        return '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, cam_id])


    def parse_img_id(self, img_id):
        return img_id.split('_')


    def get_image_path(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        img_path = os.path.join(self.root_dir, 'raw_images', sequence_name, 't0{}'.format(frame_id), 'k{}.color.jpg'.format(cam_id))
        return img_path


    def get_object_coor_path(self, img_id):
        image_path = self.get_image_path(img_id)
        coor_path = image_path.replace('raw_images', 'object_coor_maps_extend').replace('color.jpg', 'obj_coor.pkl')
        return coor_path


    def get_object_full_mask_path(self, img_id,):
        image_path = self.get_image_path(img_id)
        mask_path = image_path.replace('raw_images', 'object_coor_maps_extend').replace('color', 'mask_full')

        return mask_path


    def get_person_mask_path(self, img_id, for_aug=False):
        img_path = self.get_image_path(img_id)
        mask_path = img_path.replace('raw_images', 'person_mask').replace('color', 'person_mask')

        return mask_path


    def get_sequence_name(self, day_id, sub_id, obj_name, inter_type):
        if inter_type == 'none':
            sequence_name = 'Date0{}_Sub0{}_{}'.format(day_id, sub_id, obj_name)
        else:
            sequence_name = 'Date0{}_Sub0{}_{}_{}'.format(day_id, sub_id, obj_name, inter_type)
        return sequence_name


    def load_annotations(self, sequence_name):
        object_anno_file = os.path.join(self.root_dir, 'behave-30fps-params-v1', sequence_name, 'object_fit_all.npz')
        smpl_anno_file = os.path.join(self.root_dir, 'behave-30fps-params-v1', sequence_name, 'smpl_fit_all.npz')
        object_fit = np.load(object_anno_file)
        smpl_fit = np.load(smpl_anno_file)

        annotations = {}
        for idx, frame_name in enumerate(smpl_fit['frame_times']):
            annotations[frame_name] = {
                'poses': smpl_fit['poses'][idx],
                'betas': smpl_fit['betas'][idx],
                'trans': smpl_fit['trans'][idx],
            }
        for idx, frame_name in enumerate(object_fit['frame_times']):
            if frame_name not in annotations:
                annotations[frame_name] = {}
            annotations[frame_name]['ob_pose'] = object_fit['angles'][idx]
            annotations[frame_name]['ob_trans'] = object_fit['trans'][idx]

        return annotations


    def load_object_RT(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)

        sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        if sequence_name not in self.annotations:
            print('loading annotations for {}'.format(sequence_name))
            self.annotations[sequence_name] = self.load_annotations(sequence_name)

        annotation = self.annotations[sequence_name]['t0' + frame_id]
        obj_axis_angle = annotation['ob_pose']
        obj_rotmat = R.from_rotvec(obj_axis_angle).as_matrix()
        obj_trans = annotation['ob_trans']

        return obj_rotmat, obj_trans


    def load_smpl_params(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)
        sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        if sequence_name not in self.annotations:
            print('loading annotations for {}'.format(sequence_name))
            self.annotations[sequence_name] = self.load_annotations(sequence_name)

        annotation = self.annotations[sequence_name]['t0' + frame_id]
        smpl_params = {k: v for k, v in annotation.items() if 'ob_' not in k}
        return smpl_params


    def get_sub_gender(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)
        return self.SUBID_GENDER[sub_id]


    def get_pred_coor_map_path(self, img_id):
        image_path = self.get_image_path(img_id)
        coor_path = image_path.replace('raw_images', 'epro_pnp_extend').replace('color.jpg', 'obj_coor.pkl')
        return coor_path


    def get_openpose_path(self, img_id):
        image_path = self.get_image_path(img_id)
        openpose_path = image_path.replace('raw_images', 'openpose_extend').replace('color.jpg', 'color_keypoints.json')
        return openpose_path


    def get_obj_visible_ratio(self, img_id):
        object_full_mask = cv2.imread(self.get_object_full_mask_path(img_id), cv2.IMREAD_GRAYSCALE)
        object_full_mask = cv2.resize(object_full_mask, (0, 0),  fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        person_mask = cv2.imread(self.get_person_mask_path(img_id), cv2.IMREAD_GRAYSCALE)
        try:
            object_occlusion_mask = object_full_mask.astype(np.bool_) & person_mask.astype(np.bool_)
            visible_ratio = 1 - np.sum(object_occlusion_mask != 0) / np.sum(object_full_mask != 0)
        except: # mask may not exists
            print('Exception occurs during loading masks.')
            visible_ratio = 0.
        return visible_ratio


    def in_train_set(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)
        sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        return sequence_name not in self.dataset_splits['test']
