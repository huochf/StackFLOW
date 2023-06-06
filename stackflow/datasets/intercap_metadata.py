import os
import trimesh
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from stackflow.datasets.utils import load_pickle, load_json


class InterCapMetaData():

    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.IMG_HEIGHT = 1080
        self.IMG_WIDTH = 1920
        self.OBJECT_NAME2IDX = {
            'suitcase1': '01',
            'skate': '02',
            'sports': '03',
            'umbrella': '04',
            'tennis': '05',
            'suitcase2': '06',
            'chair1': '07',
            'bottle': '08',
            'cup': '09',
            'chair2': '10',
        }
        self.OBJECT_IDX2NAME = {v: k for k, v in self.OBJECT_NAME2IDX.items()}
        self.OBJECT_NAME2LABEL = {
            'suitcase1': 0,
            'skate': 1,
            'sports': 2,
            'umbrella': 3,
            'tennis': 4,
            'suitcase2': 5,
            'chair1': 6,
            'bottle': 7,
            'cup': 8,
            'chair2': 9,
        }
        self.OBJECT_LABEL2NAME = {v: k for k, v in self.OBJECT_NAME2LABEL.items()}
        self.SUBID_GENDER = {
            '01': 'male',
            '02': 'male',
            '03': 'female',
            '04': 'male',
            '05': 'male',
            '06': 'female',
            '07': 'female',
            '08': 'female',
            '09': 'male',
            '10': 'male',
        }
        self.OBJECT_COORDINATE_NORM = {
            'suitcase1': [0.1464, 0.2358, 0.5782],
            'skate': [0.3900, 0.0899, 0.1614],
            'sports': [0.1105, 0.1111, 0.1106],
            'umbrella': [0.4967, 0.5819, 0.4999],
            'tennis': [0.1520, 0.4692, 0.0152],
            'suitcase2': [0.2032, 0.1262, 0.1124],
            'chair1': [0.3256, 0.3633, 0.3736],
            'bottle': [0.0318, 0.0357, 0.1425],
            'cup': [0.0449, 0.0452, 0.1182],
            'chair2': [0.3120, 0.2666, 0.3079],
        }
        self.annotations = {}
        self.cam_calibration = self.load_cam_calibration()
        self.obj_mesh_templates = self.load_object_mesh_templates()
        self.obj_keypoints_dict = self.load_object_keypoints_dict()

        self.object_max_keypoint_num = 16
        self.object_num_keypoints = {
            'suitcase1': 12,
            'skate': 11,
            'sports': 1,
            'umbrella': 12,
            'tennis': 7,
            'suitcase2': 12,
            'chair1': 16,
            'bottle': 2,
            'cup': 2,
            'chair2': 10,        
        }
        self.dataset_splits = load_json('data/datasets/intercap_split.json')
        self.object_max_vertices_num = 2500
        self.object_num_vertices = {
            'suitcase1': 591,
            'skate': 1069,
            'sports': 998,
            'umbrella': 1267,
            'tennis': 2453,
            'suitcase2': 1521,
            'chair1': 342,
            'bottle': 1267,
            'cup': 1469,
            'chair2': 1002,
        }


    def go_through_all_frames(self, split='all'):
        if split == 'all':
            all_sequences = self.dataset_splits['train'] + self.dataset_splits['test']
        elif split == 'train':
            all_sequences = self.dataset_splits['train']
        elif split == 'test':
            all_sequences = self.dataset_splits['test']
        else:
            assert False

        sub_ids = sorted(os.listdir(os.path.join(self.root_dir, 'RGBD_Images')))
        for sub_id in sub_ids:
            for obj_id in os.listdir(os.path.join(self.root_dir, 'RGBD_Images', sub_id)):
                for seq_name in os.listdir(os.path.join(self.root_dir, 'RGBD_Images', sub_id, obj_id)):
                    if 'Seg' not in seq_name:
                        continue
                    seq_id = seq_name[-1]
                    seq_full_id = '_'.join([sub_id, obj_id, seq_id])
                    if seq_full_id not in all_sequences:
                        continue
                    for cam_name in os.listdir(os.path.join(self.root_dir, 'RGBD_Images', sub_id, obj_id, seq_name)):
                        for image_name in os.listdir(os.path.join(self.root_dir, 'RGBD_Images', sub_id, obj_id, seq_name, cam_name, 'color')):
                            seq_id = seq_name[-1]
                            cam_id = cam_name[-1]
                            frame_id = image_name.split('.')[0]
                            img_id = '_'.join([sub_id, obj_id, seq_id, cam_id, frame_id])
                            yield img_id


    def go_through_all_sequences(self, split='all'):
        if split == 'all':
            all_sequences = self.dataset_splits['train'] + self.dataset_splits['test']
        elif split == 'train':
            all_sequences = self.dataset_splits['train']
        elif split == 'test':
            all_sequences = self.dataset_splits['test']
        else:
            assert False

        for seq_full_id in all_sequences:
            yield seq_full_id


    def get_all_image_by_sequence(self, split='all'):
        all_sequences = list(self.go_through_all_sequences(split))
        img_ids_by_seq = {}
        for sequence_name in all_sequences:
            img_ids_by_seq[sequence_name] = {}

            sub_id, obj_id, seq_id = sequence_name.split('_')
            seq_name = 'Seg_{}'.format(seq_id)
            for cam_name in os.listdir(os.path.join(self.root_dir, 'RGBD_Images', sub_id, obj_id, seq_name)):
                frame_list = sorted(os.listdir(os.path.join(self.root_dir, 'RGBD_Images', sub_id, obj_id, seq_name, cam_name, 'color')))

                img_ids = []
                cam_id = cam_name[-1]
                for frame_name in frame_list:
                    frame_id = frame_name.split('.')[0]
                    img_id = '_'.join([sub_id, obj_id, seq_id, cam_id, frame_id])
                    img_ids.append(img_id)

                img_ids_by_seq[sequence_name][cam_id] = img_ids
        return img_ids_by_seq


    def parse_img_id(self, img_id):
        return img_id.split('_')


    def parse_object_name(self, img_id):
        sub_id, obj_id, seq_id, cam_id, frame_id = img_id.split('_')
        obj_name = self.OBJECT_IDX2NAME[obj_id]
        return obj_name


    def get_sequence_dir(self, seq_full_id):
        sub_id, obj_id, seq_id = seq_full_id.split('_')
        seq_dir = os.path.join(self.root_dir, 'RGBD_Images', sub_id, obj_id, 'Seg_{}'.format(seq_id))
        return seq_dir


    def get_annotations_file(self, seq_full_id):
        seq_dir = self.get_sequence_dir(seq_full_id)
        seq_dir = seq_dir.replace('RGBD_Images', 'Res_tuned')
        return os.path.join(seq_dir, 'res_tuned.pkl')


    def load_object_mesh_templates(self, ):
        templates = {}
        for object_label in range(10):
            object_id = '{:02d}'.format(object_label + 1)
            object_mesh = trimesh.load(os.path.join(self.root_dir, 'objs', '{}.ply'.format(object_id)), process=False)
            vertices = np.array(object_mesh.vertices, dtype=np.float32)
            vertices = vertices - vertices.mean(axis=0).reshape(1, 3)
            faces = np.array(object_mesh.faces, dtype=np.int64)
            object_name = self.OBJECT_IDX2NAME[object_id]
            templates[object_name] = (vertices, faces)

        return templates


    def load_object_trimesh(self, ):
        templates = {}
        for object_label in range(10):
            object_id = '{:02d}'.format(object_label + 1)
            object_mesh = trimesh.load(os.path.join(self.root_dir, 'objs', '{}.ply'.format(object_id)), process=False)
            vertices = np.array(object_mesh.vertices, dtype=np.float32)
            object_mesh.vertices = vertices - vertices.mean(axis=0).reshape(1, 3)
            object_name = self.OBJECT_IDX2NAME[object_id]
            templates[object_name] = object_mesh

        return templates


    def load_object_keypoints_dict(self, ):
        keypoints_dict = {}
        for object_name in self.OBJECT_NAME2IDX.keys():
            keypoints_path = os.path.join('./data/datasets/intercap_obj_keypoints/{}_keypoints.json'.format(object_name))
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


    def load_object_RT(self, img_id):
        sub_id, obj_id, seq_id, cam_id, frame_id = img_id.split('_')
        seq_id = '_'.join([sub_id, obj_id, seq_id])
        if seq_id not in self.annotations:
            annotation_file = self.get_annotations_file(seq_id)
            print('loading {}'.format(annotation_file))
            self.annotations[seq_id] = load_pickle(annotation_file)

        annotation = self.annotations[seq_id][int(frame_id)]
        obj_axis_angle = annotation['ob_pose']
        obj_rotmat = Rotation.from_rotvec(obj_axis_angle).as_matrix()
        obj_trans = annotation['ob_trans']

        return obj_rotmat, obj_trans


    def load_smpl_params(self, img_id):
        sub_id, obj_id, seq_id, cam_id, frame_id = img_id.split('_')
        seq_id = '_'.join([sub_id, obj_id, seq_id])
        if seq_id not in self.annotations:
            annotation_file = self.get_annotations_file(seq_id)
            print('loading {}'.format(annotation_file))
            self.annotations[seq_id] = load_pickle(annotation_file)
        annotation = self.annotations[seq_id][int(frame_id)]
        smpl_params = {k: v for k, v in annotation.items() if 'ob_' not in k}
        return smpl_params


    def load_cam_calibration(self, ):
        cam_calibration = {}
        for cam_id in range(6):
            if cam_id == 0:
                cam_params = load_json(os.path.join(self.root_dir, 'Data/calibration_third/Color.json'))
            else:
                cam_params = load_json(os.path.join(self.root_dir, 'Data/calibration_third/Color_{}.json').format(cam_id + 1))
            cam_calibration[str(cam_id + 1)] = cam_params
        return cam_calibration


    def get_image_path(self, img_id):
        sub_id, obj_id, seq_id, cam_id, frame_id = img_id.split('_')
        image_path = os.path.join(self.root_dir, 'RGBD_Images', sub_id, obj_id, 'Seg_{}'.format(seq_id), 'Frames_Cam{}'.format(cam_id), 'color', '{}.jpg'.format(frame_id))
        return image_path


    def get_person_mask_path(self, img_id):
        image_path = self.get_image_path(img_id)
        person_mask_path = image_path.replace('RGBD_Images', 'mask').replace('color', 'mask').replace('.jpg', '_person.jpg')
        return person_mask_path


    def get_object_full_mask_path(self, img_id):
        image_path = self.get_image_path(img_id)
        obj_mask_path = image_path.replace('RGBD_Images', 'object_coor_maps').replace('color', 'obj_full_mask')
        return obj_mask_path


    def get_object_coor_path(self, img_id):
        image_path = self.get_image_path(img_id)
        obj_coor_path = image_path.replace('RGBD_Images', 'object_coor_maps').replace('color', 'obj_coor').replace('.jpg', '.pkl')
        return obj_coor_path


    def get_pred_coor_map_path(self, img_id):
        image_path = self.get_image_path(img_id)
        obj_coor_path = image_path.replace('RGBD_Images', 'epro_pnp').replace('color', 'obj_coor').replace('.jpg', '.pkl')
        return obj_coor_path


    def get_openpose_path(self, img_id):
        image_path = self.get_image_path(img_id)
        openpose_path = image_path.replace('RGBD_Images', 'openpose').replace('.jpg', '_keypoints.json')
        return openpose_path


    def get_gt_meshes(self, img_id):
        sub_id, obj_id, seq_id, cam_id, frame_id = img_id.split('_')
        seq_dir = os.path.join(self.root_dir, 'Res', sub_id, obj_id, 'Seg_{}'.format(seq_id), 'Mesh')
        smpl_mesh_path = os.path.join(seq_dir, '{}_second.ply'.format(frame_id))
        object_mesh_path = os.path.join(seq_dir, '{}_second_obj.ply'.format(frame_id))

        calitration = self.cam_calibration[cam_id]
        cam_R = np.array(calitration['R'])
        cam_R = Rotation.from_rotvec(cam_R).as_matrix()
        cam_T = np.array(calitration['T'])

        smpl_mesh = trimesh.load(smpl_mesh_path, process=False)
        smpl_mesh.vertices = np.matmul(np.array(smpl_mesh.vertices), cam_R.T) + cam_T.reshape(1, 3)
        object_mesh = trimesh.load(object_mesh_path, process=False)
        object_mesh.vertices = np.matmul(np.array(object_mesh.vertices), cam_R.T) + cam_T.reshape(1, 3)

        return smpl_mesh, object_mesh


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
        sub_id, obj_id, seq_id, cam_id, frame_id = img_id.split('_')
        seq_full_id = '_'.join([sub_id, obj_id, seq_id])
        return seq_full_id in self.dataset_splits['train']
