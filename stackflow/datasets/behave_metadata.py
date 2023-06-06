import os
import trimesh
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from stackflow.datasets.utils import load_json, load_pickle


class BEHAVEMetaData():

    def __init__(self, root_dir):
        
        self.root_dir = root_dir
        self.IMG_HEIGHT = 1536
        self.IMG_WIDTH = 2048
        self.OBJECT_NAME_MAP = {'basketball': 'sports ball', 
                                'chairblack': 'chair',
                                'chairwood': 'chair',
                                'yogaball': 'sports ball',
                                'chairblack': 'chair',}
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
        self.AVATAR_NAME2ID = {
            '00032_shortlong': '01',
            '00032_shortshort': '02',
            '00096_shortlong': '03',
            '00096_shortshort': '04',
            '00159_shortlong': '05',
            '00159_shortshort': '06',
            '03223_shortlong': '07',
            '03223_shortshort': '08',
        }
        self.AVATAR_ID2NAME = {v: k for k, v in self.AVATAR_NAME2ID.items()}

        self.dataset_splits = load_json(os.path.join(root_dir, 'split.json'))
        self.cam_intrinsics = self.load_cam_intrinsics()
        self.cam_RT_matrix = self.load_cam_RT_matrix()
        self.cam_intrinsics_aug = (1018.952, 779.486, 979.7844, 979.840)

        self.obj_mesh_templates = self.load_object_mesh_templates()
        self.obj_keypoints_dict = self.load_object_keypoints_dict()

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


    def parse_seq_info(self, sequence_name):
        try:
            day_name, sub_name, obj_name, inter_type = sequence_name.split('_')
        except:
            day_name, sub_name, obj_name = sequence_name.split('_')
            inter_type = 'none'
        day_id, sub_id = day_name[5:], sub_name[4:]

        return day_id, sub_id, obj_name, inter_type


    def go_through_all_frames(self, split='all'):
        if split == 'all':
            sequences = os.listdir(os.path.join(self.root_dir, 'sequences'))
        elif split == 'train':
            sequences = self.dataset_splits['train']
        elif split == 'test':
            sequences = self.dataset_splits['test']
        else:
            sequences = []

        for sequence_name in sequences:
            day_id, sub_id, obj_name, inter_type = self.parse_seq_info(sequence_name)
            for frame_name in os.listdir(os.path.join(self.root_dir, 'sequences', sequence_name)):
                if frame_name == 'info.json':
                    continue
                frame_id = frame_name[2:5]
                for cam_id in range(4):
                    img_id = self.get_img_id(day_id, sub_id, obj_name, inter_type, frame_id, str(cam_id))
                    yield img_id


    def go_through_all_frames_aug(self, ):
        sequences = os.listdir(os.path.join(self.root_dir, 'rendered_images'))
        for sequence_name in sequences:
            day_id, sub_id, obj_name, inter_type = self.parse_seq_info(sequence_name)
            for frame_name in os.listdir(os.path.join(self.root_dir, 'rendered_images', sequence_name)):
                frame_id = frame_name[2:5]
                for avatar_name in os.listdir(os.path.join(self.root_dir, 'rendered_images', sequence_name, frame_name)):
                    avatar_id = self.AVATAR_NAME2ID[avatar_name]
                    for cam_id in range(12):
                        img_id = '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, avatar_id, '{:02d}'.format(cam_id)])
                        yield img_id



    def get_img_id(self, day_id, sub_id, obj_name, inter_type, frame_id, cam_id):
        return '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, cam_id])


    def parse_img_id(self, img_id):
        return img_id.split('_')


    def parse_object_name(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        return obj_name


    def get_sequence_name(self, day_id, sub_id, obj_name, inter_type):
        if inter_type == 'none':
            sequence_name = 'Date0{}_Sub0{}_{}'.format(day_id, sub_id, obj_name)
        else:
            sequence_name = 'Date0{}_Sub0{}_{}_{}'.format(day_id, sub_id, obj_name, inter_type)
        return sequence_name


    def get_frame_dir(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)
        sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        frame_dir = os.path.join(self.root_dir, 'sequences', sequence_name, 't0{}.000'.format(frame_id), )
        return frame_dir


    def load_object_RT(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)
        frame_dir = self.get_frame_dir(img_id)
        object_param_path = os.path.join(frame_dir, obj_name, 'fit01', '{}_fit.pkl'.format(obj_name))
        if not os.path.exists(object_param_path):
            _obj_name = self.OBJECT_NAME_MAP[obj_name]
            object_param_path = os.path.join(frame_dir, _obj_name, 'fit01', '{}_fit.pkl'.format(_obj_name))
        object_param = load_pickle(object_param_path)
        object_trans = object_param['trans']
        object_pose = object_param['angle']
        object_rotmat = R.from_rotvec(object_pose).as_matrix()

        cam_R, cam_T = self.cam_RT_matrix[day_id][int(cam_id)]
        object_trans = np.matmul(cam_R.transpose(), object_trans - cam_T).reshape(3, )
        object_rotmat = np.matmul(cam_R.transpose(), object_rotmat)
        return object_rotmat, object_trans


    def load_object_RT_aug(self, img_id):
        param_path = self.get_image_path(img_id, for_aug=True).replace('hoi_color.jpg', 'params.json')
        params = load_json(param_path)

        object_rotmat = np.array(params['object_RT']['R'], dtype=np.float32).reshape((3, 3))
        object_trans = np.array(params['object_RT']['T'], dtype=np.float32).reshape(3, )
        return object_rotmat, object_trans


    def load_smpl_params(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)

        frame_dir = self.get_frame_dir(img_id)
        smplh_params = load_pickle(os.path.join(frame_dir, 'person', 'fit02', 'person_fit.pkl'))

        cam_R, cam_T = self.cam_RT_matrix[day_id][int(cam_id)]
        smplh_trans = smplh_params['trans']
        smplh_params['trans'] = np.matmul(cam_R.transpose(), smplh_trans - cam_T)

        smplh_pose = smplh_params['pose']
        global_pose = smplh_pose[:3]
        global_pose_rotmat = R.from_rotvec(global_pose).as_matrix()
        global_pose_rotmat = np.matmul(cam_R.transpose(), global_pose_rotmat)
        global_pose = R.from_matrix(global_pose_rotmat).as_rotvec()
        smplh_params['pose'][:3] = global_pose

        return smplh_params


    def load_smpl_params_aug(self, img_id):
        param_path = self.get_image_path(img_id, for_aug=True).replace('hoi_color.jpg', 'params.json')
        params = load_json(param_path)

        smplh_params = {}
        smplh_params['betas'] = np.array(params['smplh_params']['betas'], dtype=np.float32).reshape(10, )
        smplh_params['pose'] = np.array(params['smplh_params']['pose'], dtype=np.float32).reshape(72, )
        smplh_params['trans'] = np.array(params['smplh_params']['transl'], dtype=np.float32).reshape(3, )

        return smplh_params


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


    def get_image_path(self, img_id, for_aug=False):
        if not for_aug:
            day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)
            frame_dir = self.get_frame_dir(img_id)
            return os.path.join(frame_dir, 'k{}.color.jpg'.format(cam_id))
        else:
            day_id, sub_id, obj_name, inter_type, frame_id, avatar_id, cam_id = img_id.split('_')
            sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
            img_path = os.path.join(self.root_dir, 'rendered_images', sequence_name, 't0{}.000'.format(frame_id), 
                self.AVATAR_ID2NAME[avatar_id], 'k{}_hoi_color.jpg'.format(cam_id))
            return img_path


    def get_person_mask_path(self, img_id, for_aug=False):
        img_path = self.get_image_path(img_id, for_aug)
        if not for_aug:
            mask_path = img_path.replace('color', 'person_mask')
        else:  
            mask_path = img_path.replace('hoi_color', 'person_mask')

        return mask_path


    def get_object_coor_path(self, img_id):
        image_path = self.get_image_path(img_id)
        coor_path = image_path.replace('sequences', 'object_coor_maps').replace('color.jpg', 'obj_coor.pkl')
        return coor_path


    def get_object_render_mask_path(self, img_id):
        image_path = self.get_image_path(img_id)
        mask_path = image_path.replace('color', 'obj_rend_mask')
        return mask_path


    def get_object_full_mask_path(self, img_id, for_aug=False):
        if not for_aug:
            image_path = self.get_image_path(img_id)
            mask_path = image_path.replace('sequences', 'object_coor_maps').replace('color', 'mask_full')
        else:
            image_path = self.get_image_path(img_id, for_aug)
            mask_path = image_path.replace('hoi_color', 'object_mask')

        return mask_path


    def get_pred_coor_map_path(self, img_id):
        image_path = self.get_image_path(img_id)
        coor_path = image_path.replace('sequences', 'epro_pnp').replace('color.jpg', 'obj_coor.pkl')
        return coor_path


    def get_openpose_path(self, img_id):
        image_path = self.get_image_path(img_id)
        openpose_path = image_path.replace('sequences', 'openpose').replace('color.jpg', 'color_keypoints.json')
        return openpose_path


    def get_sub_gender(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)
        return self.SUBID_GENDER[sub_id]


    def get_gt_meshes(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        frame_dir = self.get_frame_dir(img_id)
        smpl_gt_mesh_path = os.path.join(frame_dir, 'person', 'fit02', 'person_fit.ply')
        obj_gt_mesh_path = os.path.join(frame_dir, obj_name, 'fit01', '{}_fit.ply'.format(obj_name))
        if not os.path.exists(obj_gt_mesh_path):
            _obj_name = self.OBJECT_NAME_MAP[obj_name]
            obj_gt_mesh_path = os.path.join(frame_dir, _obj_name, 'fit01', '{}_fit.ply'.format(_obj_name))

        cam_R, cam_T = self.cam_RT_matrix[day_id][int(cam_id)]
        smpl_gt_mesh = trimesh.load(smpl_gt_mesh_path, process=False)
        smpl_gt_mesh.vertices = np.matmul(smpl_gt_mesh.vertices - cam_T.reshape(1, 3), cam_R)

        obj_gt_mesh = trimesh.load(obj_gt_mesh_path, process=False)
        obj_gt_mesh.vertices = np.matmul(obj_gt_mesh.vertices - cam_T.reshape(1, 3), cam_R)

        return smpl_gt_mesh, obj_gt_mesh


    def get_obj_visible_ratio(self, img_id):
        object_full_mask = cv2.imread(self.get_object_full_mask_path(img_id), cv2.IMREAD_GRAYSCALE)
        object_full_mask = cv2.resize(object_full_mask, (0, 0),  fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        object_render_mask = cv2.imread(self.get_object_render_mask_path(img_id), cv2.IMREAD_GRAYSCALE)
        try:
            visible_ratio = np.sum(object_render_mask > 127) / np.sum(object_full_mask > 127)
        except: # mask may not exists
            print('Exception occurs during loading masks.')
            visible_ratio = 1.
        return visible_ratio


    def in_train_set(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)
        sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        return sequence_name not in self.dataset_splits['test']
