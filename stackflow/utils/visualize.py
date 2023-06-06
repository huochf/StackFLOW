import os
import numpy as np
import cv2
import torch
import neural_renderer as nr

from stackflow.datasets.utils import load_pickle



def visualize_step(cfg, dataset_metadata, batch, pred, epoch, idx):

    output_dir = os.path.join(cfg.train.output_dir, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    image = batch['image']
    image_show = image.permute(0, 2, 3, 1).cpu().numpy()
    mean = np.array(cfg.dataset.mean).reshape(1, 1, 1, 3)
    std = np.array(cfg.dataset.std).reshape(1, 1, 1, 3)
    image_show = image_show * std + mean

    scale = cfg.dataset.img_size
    center = np.array([cfg.dataset.img_size / 2, cfg.dataset.img_size / 2]).reshape(1, 1, 2)
    pred_joint_2d = pred['pred_joint_2d'].detach().cpu().numpy() * scale + center
    pred_obj_kps_2d = pred['object_keypoints_2d'].detach().cpu().numpy() * scale + center
    gt_joint_2d = batch['person_joint_2d'].detach().cpu().numpy() * scale + center
    gt_obj_kps_2d = batch['object_kpts_2d'].detach().cpu().numpy() * scale + center

    center = np.array([cfg.dataset.img_size / 2, cfg.dataset.img_size / 2]).reshape(1, 1, 1, 2)
    pred_joint_2d_samples = pred['pred_joint_2d_samples'].detach().cpu().numpy() * scale + center
    pred_obj_kps_2d_samples = pred['object_keypoints_2d_samples'].detach().cpu().numpy() * scale + center

    object_labels = batch['object_labels'].cpu().numpy()
    obj_names = list(dataset_metadata.OBJECT_NAME2IDX.keys())
    batch_size = image_show.shape[0]
    for i in range(batch_size):
        obj_name = obj_names[object_labels[i]]
        num_kps = dataset_metadata.object_num_keypoints[obj_name]
        image = image_show[i].copy()
        image = draw_smpl_joints(image, pred_joint_2d[i])
        image = draw_object_keypoints(image, pred_obj_kps_2d[i][:num_kps], obj_name)

        cv2.imwrite(os.path.join(output_dir, '{}_{}_{}_pred_kps.jpg'.format(epoch, idx, i)), image[:, :, ::-1])

        image = image_show[i].copy()
        image = draw_smpl_joints(image, gt_joint_2d[i])
        image = draw_object_keypoints(image, gt_obj_kps_2d[i][:num_kps], obj_name)

        cv2.imwrite(os.path.join(output_dir, '{}_{}_{}_gt_kps.jpg'.format(epoch, idx, i)), image[:, :, ::-1])

        num_samples = pred_joint_2d_samples.shape[1]
        for j in range(num_samples):
            image = image_show[i].copy()
            image = draw_smpl_joints(image, pred_joint_2d_samples[i, j])
            image = draw_object_keypoints(image, pred_obj_kps_2d_samples[i, j][:num_kps], obj_name)

            cv2.imwrite(os.path.join(output_dir, '{}_{}_{}_{}_pred_kps.jpg'.format(epoch, idx, i, j)), image[:, :, ::-1])



def render_hoi(image, smpl_v, smpl_f, object_v, object_f, K):
    device = torch.device('cuda')
    smpl_v = torch.tensor(smpl_v, dtype=torch.float32).reshape(1, -1, 3).to(device)
    object_v = torch.tensor(object_v, dtype=torch.float32).reshape(1, -1, 3).to(device)
    smpl_f = torch.tensor(smpl_f, dtype=torch.int64).reshape(1, -1, 3).to(device)
    object_f = torch.tensor(object_f, dtype=torch.int64).reshape(1, -1, 3).to(device)

    vertices = torch.cat([smpl_v, object_v], dim=1)
    faces = torch.cat([smpl_f, object_f + smpl_v.shape[1]], dim=1)

    colors_list = [
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # red
        [0.65098039, 0.74117647, 0.85882353],  # blue
        [0.9, 0.7, 0.7],  # pink
    ]
    smpl_t = torch.tensor(colors_list[1], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, smpl_f.shape[1], 1, 1, 1, 1)
    object_t = torch.tensor(colors_list[0], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, object_f.shape[1], 1, 1, 1, 1)
    textures = torch.cat([smpl_t, object_t], dim=1).to(device)

    K = torch.tensor(K, dtype=torch.float32).reshape(1, 3, 3).to(device)
    R = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).to(device)
    t = torch.zeros(3, dtype=torch.float32).reshape(1, 3).to(device)

    h, w, _ = image.shape
    s = max(h, w)
    renderer = nr.renderer.Renderer(image_size=s, K=K, R=R, t=t, orig_size=s)
    
    renderer.background_color = [1, 1, 1]
    renderer.light_direction = [1, 0.5, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5

    rend, _, mask = renderer.render(vertices=vertices, faces=faces, textures=textures)
    rend = rend.clip(0, 1)

    rend = rend[0, :, :h, :w,].permute(1, 2, 0).detach().cpu().numpy()
    mask = mask[0, :h, :w, ].detach().cpu().numpy().astype(np.bool)

    image[mask] = rend[mask] * 255

    return image


def render_multi_hoi(image, smpl_v, smpl_f, object_v_list, object_f_list, K):
    device = torch.device('cuda')
    smpl_v = torch.tensor(smpl_v, dtype=torch.float32).reshape(1, -1, 3).to(device)
    smpl_f = torch.tensor(smpl_f, dtype=torch.int64).reshape(1, -1, 3).to(device)
    object_v_list = [torch.tensor(object_v, dtype=torch.float32).reshape(1, -1, 3).to(device) for object_v in object_v_list]
    object_f_list = [torch.tensor(object_f, dtype=torch.int64).reshape(1, -1, 3).to(device) for object_f in object_f_list]

    colors_list = [
        [0.65098039, 0.74117647, 0.85882353],  # blue
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # red
        [60 / 255., 179 / 255., 113 / 255.], # SpringGreen
        [244 / 255., 164 / 255., 96 / 255.], # SandyBrown
        [0.9, 0.7, 0.7],  # pink
    ]

    vertices = smpl_v
    faces = smpl_f
    textures = torch.tensor(colors_list[0], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, smpl_f.shape[1], 1, 1, 1, 1).to(device)
    for idx, (object_v, object_f) in enumerate(zip(object_v_list, object_f_list)):
        faces = torch.cat([faces, object_f + vertices.shape[1]], dim=1)
        vertices = torch.cat([vertices, object_v], dim=1)
        textures = torch.cat([textures, torch.tensor(colors_list[idx + 1], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, object_f.shape[1], 1, 1, 1, 1).to(device)], dim=1)

    K = torch.tensor(K, dtype=torch.float32).reshape(1, 3, 3).to(device)
    R = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).to(device)
    t = torch.zeros(3, dtype=torch.float32).reshape(1, 3).to(device)

    h, w, _ = image.shape
    s = max(h, w)
    renderer = nr.renderer.Renderer(image_size=s, K=K, R=R, t=t, orig_size=s)
    
    renderer.background_color = [1, 1, 1]
    renderer.light_direction = [1, 0.5, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5

    rend, _, mask = renderer.render(vertices=vertices, faces=faces, textures=textures)
    rend = rend.clip(0, 1)

    rend = rend[0, :, :h, :w,].permute(1, 2, 0).detach().cpu().numpy()
    mask = mask[0, :h, :w, ].detach().cpu().numpy().astype(np.bool)

    image[mask] = rend[mask] * 255

    return image


def get_offset_heatmap(distances, anchor_indices, vertices):
    vertices = vertices.reshape(-1, 3)
    anchor_indices = anchor_indices.reshape(-1)
    distances = distances.reshape(-1)

    colors_list = [
        [255 / 225., 255 / 255., 49 / 255.], # yellow
        [1., 0., 0.], # red
        [0., 0., 1.], # blue
    ]
    yellow = torch.tensor(colors_list[0]).float().to(distances.device).reshape(1, 3)
    red = torch.tensor(colors_list[1]).float().to(distances.device).reshape(1, 3)
    blue = torch.tensor(colors_list[2]).float().to(distances.device).reshape(1, 3)
    min_distances = distances.min()
    scale = 15
    # distances = (distances - distances.min()) / (distances.max() - distances.min()) * scale
    distances = (distances - distances.min()) * scale
    distances += max(0, min_distances - 0.02)

    thresh = 0.1 * scale
    color_weight1 = ((distances.exp() - 1) / (np.exp(thresh) - 1)).reshape(-1, 1)
    color_weight2 = ((thresh - distances).exp()).reshape(-1, 1)
    anchor_colors = (1 - color_weight1) * yellow + color_weight1 * red
    anchor_colors[distances > thresh] = (color_weight2 * red + (1 - color_weight2) * blue)[distances > thresh]
    # anchor_colors = ((1 - color_weight2) * red + color_weight2 * blue)
    # print(color_weight2.min(), color_weight2.max())
    weights = ((- ((vertices.unsqueeze(1) - vertices[anchor_indices].unsqueeze(0)) ** 2).sum(-1).sqrt()).exp() * 8).softmax(-1)
    colors = weights @ anchor_colors
    return colors


def render_multi_hoi_with_offsets(image, pred_offsets, objects, smpl_v, smpl_f, object_v_list, object_f_list, K):
    device = torch.device('cuda')
    pred_offsets = torch.tensor(pred_offsets, dtype=torch.float32).reshape(-1, 32 * 22, 64, 3).to(device)
    pred_offsets = (pred_offsets ** 2).sum(-1).sqrt()

    smpl_v = torch.tensor(smpl_v, dtype=torch.float32).reshape(1, -1, 3).to(device)
    smpl_f = torch.tensor(smpl_f, dtype=torch.int64).reshape(1, -1, 3).to(device)
    object_v_list = [torch.tensor(object_v, dtype=torch.float32).reshape(1, -1, 3).to(device) for object_v in object_v_list]
    object_f_list = [torch.tensor(object_f, dtype=torch.int64).reshape(1, -1, 3).to(device) for object_f in object_f_list]

    colors_list = [
        [0.65098039, 0.74117647, 0.85882353],  # blue
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # red
        [60 / 255., 179 / 255., 113 / 255.], # SpringGreen
        [244 / 255., 164 / 255., 96 / 255.], # SandyBrown
        [0.9, 0.7, 0.7],  # pink
    ]
    pca_models = load_pickle('data/datasets/behave_pca_models_n32_64_d32.pkl')
    smpl_anchor_indices = torch.tensor(pca_models['backpack']['smpl_anchor_indices'], dtype=torch.int64).to(device)
    smpl_min_distances = pred_offsets.min(2)[0].min(0)[0].reshape(-1)
    smpl_v_colors = get_offset_heatmap(smpl_min_distances, smpl_anchor_indices, smpl_v)
    object_v_color_list = []
    for i, (obj_name, object_v) in enumerate(zip(objects, object_v_list)):
        object_min_distance = pred_offsets[i].min(0)[0]
        object_anchor_indices = torch.tensor(pca_models[obj_name]['object_anchor_indices'], dtype=torch.int64).to(device)
        object_v_colors = get_offset_heatmap(object_min_distance, object_anchor_indices, object_v)
        object_v_color_list.append(object_v_colors)

    vertices = smpl_v
    faces = smpl_f
    textures = smpl_v_colors[smpl_f[0, :, 0]].reshape(1, -1, 1, 1, 1, 3)
    for idx, (object_v, object_f) in enumerate(zip(object_v_list, object_f_list)):
        faces = torch.cat([faces, object_f + vertices.shape[1]], dim=1)
        vertices = torch.cat([vertices, object_v], dim=1)
        object_textures = object_v_color_list[idx][object_f[0, :, 0]].reshape(1, -1, 1, 1, 1, 3)
        textures = torch.cat([textures, object_textures], dim=1)

    K = torch.tensor(K, dtype=torch.float32).reshape(1, 3, 3).to(device)
    R = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).to(device)
    t = torch.zeros(3, dtype=torch.float32).reshape(1, 3).to(device)

    h, w, _ = image.shape
    s = max(h, w)
    renderer = nr.renderer.Renderer(image_size=s, K=K, R=R, t=t, orig_size=s)
    
    renderer.background_color = [1, 1, 1]
    renderer.light_direction = [1, 0.5, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5

    rend, _, mask = renderer.render(vertices=vertices, faces=faces, textures=textures)
    rend = rend.clip(0, 1)

    rend = rend[0, :, :h, :w,].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
    mask = mask[0, :h, :w, ].detach().cpu().numpy().astype(np.bool)

    image[mask] = rend[mask] * 255

    return image


def render_multi_hoi_video(image, smpl_v, smpl_J, smpl_f, object_v_list, object_f_list, K, outpath):
    h, w, _ = image.shape
    video = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    for i in range(20):
        video.write(image)

    background = np.ones((h, w, 3), dtype=np.uint8) * 255
    img_hoi = render_multi_hoi(background.copy(), smpl_v, smpl_f, object_v_list, object_f_list, K)
    for alpha in np.arange(0, 1, 0.05):
        img = image * (1 - alpha) + img_hoi * alpha
        video.write(img.astype(np.uint8))

    for rot in range(0, -362, -2):
        _smpl_v = rotation(smpl_v, smpl_J[0:1], rot)
        _object_v_list = []
        for object_v in object_v_list:
            _object_v_list.append(rotation(object_v, smpl_J[0:1], rot))
        img_hoi = render_multi_hoi(background.copy(), _smpl_v, smpl_f, _object_v_list, object_f_list, K)
        video.write(img_hoi.astype(np.uint8))

    for alpha in np.arange(0, 1, 0.05):
        img = img_hoi * (1 - alpha) + image * alpha
        video.write(img.astype(np.uint8))

    video.release()


def render_multi_hoi_video_with_offsets(image, pred_offsets, objects, smpl_v, smpl_J, smpl_f, object_v_list, object_f_list, K, outpath):
    # this implementation may be not efficient.
    h, w, _ = image.shape
    video = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    for i in range(20):
        video.write(image)

    background = np.ones((h, w, 3), dtype=np.uint8) * 255
    img_hoi = render_multi_hoi_with_offsets(background.copy(), pred_offsets, objects, smpl_v, smpl_f, object_v_list, object_f_list, K)
    for alpha in np.arange(0, 1, 0.05):
        img = image * (1 - alpha) + img_hoi * alpha
        video.write(img.astype(np.uint8))

    for rot in range(0, -362, -2):
        _smpl_v = rotation(smpl_v, smpl_J[0:1], rot)
        _object_v_list = []
        for object_v in object_v_list:
            _object_v_list.append(rotation(object_v, smpl_J[0:1], rot))
        img_hoi = render_multi_hoi_with_offsets(background.copy(), pred_offsets, objects, _smpl_v, smpl_f, _object_v_list, object_f_list, K)
        video.write(img_hoi.astype(np.uint8))

    for alpha in np.arange(0, 1, 0.05):
        img = img_hoi * (1 - alpha) + image * alpha
        video.write(img.astype(np.uint8))

    video.release()


def rotation(v, center, angle):
    v_centered = v - center.reshape(1, 3)
    from scipy.spatial.transform import Rotation
    rot_matrix = Rotation.from_euler('y', angle, degrees=True).as_matrix()
    v_out = v_centered @ rot_matrix + center.reshape(1, 3)
    return v_out


def draw_smpl_joints(image, joints_2d):
    joints_2d = joints_2d[:22]

    bone_idx = [[ 0,  1], [ 0,  2], [ 0,  3], 
                [ 1,  4], [ 2,  5], [ 3,  6], 
                [ 4,  7], [ 5,  8], [ 6,  9], 
                [ 7, 10], [ 8, 11], [ 9, 12], 
                [ 9, 13], [ 9, 14], [12, 15],
                [13, 16], [14, 17], [16, 18],
                [17, 19], [18, 20], [19, 21]]
    points_colors = [
        [0.,    255.,  255.],
        [0.,   255.,    170.],
        [0., 170., 255.,],
        [85., 170., 255.],
        [0.,   255.,   85.], # 4
        [0., 85., 255.],
        [170., 85., 255.],
        [0.,   255.,   0.], # 7
        [0., 0., 255.], 
        [255., 0., 255.],
        [0.,    255.,  0.], # 10
        [0., 0., 255.],
        [255., 85., 170.],
        [170., 0, 255.],
        [255., 0., 170.],
        [255., 170., 85.],
        [85., 0., 255.],
        [255., 0., 85],
        [32., 0., 255.],
        [255., 0, 32],
        [0., 0., 255.],
        [255., 0., 0.],
    ]

    line_thickness = 2
    thickness = 4
    lineType = 8

    for bone in bone_idx:
        idx1, idx2 = bone
        x1, y1 = joints_2d[idx1]
        x2, y2 = joints_2d[idx2]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), tuple(points_colors[idx1]), line_thickness, lineType)

    for i, points in enumerate(joints_2d):
        x, y = points
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), thickness, points_colors[i], thickness=-1, lineType=lineType)

    return image


def draw_object_keypoints(image, keypoints_2d, obj_name):
    keypoint_linkages = {
        'backpack': [[0, 1],[1, 2], [2, 3], [3, 0], [0, 5], [1, 6], [2, 7], [3, 4], [4, 5], [5, 6], [6, 7], [7, 4]],
        'basketball': [],
        'boxlarge': [[0, 1],[1, 2], [2, 3], [3, 0], [0, 5], [1, 6], [2, 7], [3, 4], [4, 5], [5, 6], [6, 7], [7, 4]],
        'boxlong': [[0, 1],[1, 2], [2, 3], [3, 0], [0, 5], [1, 6], [2, 7], [3, 4], [4, 5], [5, 6], [6, 7], [7, 4]],
        'boxmedium': [[0, 1],[1, 2], [2, 3], [3, 0], [0, 5], [1, 6], [2, 7], [3, 4], [4, 5], [5, 6], [6, 7], [7, 4]],
        'boxsmall': [[0, 1],[1, 2], [2, 3], [3, 0], [0, 5], [1, 6], [2, 7], [3, 4], [4, 5], [5, 6], [6, 7], [7, 4]],
        'boxtiny': [[0, 1],[1, 2], [2, 3], [3, 0], [0, 5], [1, 6], [2, 7], [3, 4], [4, 5], [5, 6], [6, 7], [7, 4]],
        'chairblack': [[0, 1], [1, 13], [13, 2], [2, 15], [15, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 12], [12, 9],
                       [9, 10], [10, 0], [10, 11], [11, 12], [13, 14], [14, 15], [3, 8], [2, 9]],
        'chairwood': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 2], [2, 6], [3, 7], [4, 8], [5, 9], [0, 5]],
        'keyboard': [[0, 1], [1, 2], [2, 3], [3, 0]],
        'monitor': [[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4], [4, 5], [5, 6], [5, 7]],
        'plasticcontainer': [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], 
                             [7, 4], [0, 4], [1, 5], [2, 6], [3, 7], [4, 6], [5, 7]],
        'stool': [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [1, 4], [2, 5]],
        'suitcase': [[0, 1],[1, 2], [2, 3], [3, 0], [0, 5], [1, 6], [2, 7], [3, 4], [4, 5], [5, 6], [6, 7], [7, 4]],
        'tablesmall': [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4], [4, 5], [5, 6], [5, 7], [5, 8], [5, 9]],
        'tablesquare': [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7]],
        'toolbox': [[0, 1],[1, 2], [2, 3], [3, 0], [0, 5], [1, 6], [2, 7], [3, 4], [4, 5], [5, 6], [6, 7], [7, 4]],
        'trashbin': [[0, 1]],
        'yogaball': [],
        'yogamat': [[0, 1]],
        'suitcase1': [[0, 1], [1, 2], [0, 3], [2, 4], [2, 3], [3, 7], [4, 5], [6, 7], 
                      [5, 6], [4, 8], [5, 9], [6, 10], [7, 11], [8, 9], [9, 10], [10, 11], [8, 11]],
        'sports': [],
        'umbrella': [[0, 1], [1, 2], [2, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11]],
        'suitcase2': [[0, 1], [1, 2], [0, 3], [2, 3], [4, 7], [2, 4], [2, 5], [3, 6], [3, 7], [4, 5], [6, 7], 
                      [5, 6], [4, 8], [5, 9], [6, 10], [7, 11], [8, 9], [9, 10], [10, 11], [8, 11]],
        'chair1': [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 6], [1, 7], [1, 8], [1, 9], [6, 7], [7, 8], [8, 9], [9, 6],
                   [6, 10], [10, 12], [10, 11], [11, 12], [9, 13], [13, 14], [13, 15], [14, 15], [11, 14]],
        'bottle': [[0, 1]],
        'cup': [[0, 1]],
        'chair2': [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 6], [1, 7], [1, 8], [1, 9], [6, 7], [7, 8], [8, 9], [9, 6]],
        'skate': [[0, 1], [1, 6], [6, 2], [2, 0], [1, 7], [7, 5], [2, 9], [9, 4], 
                    [3, 4], [4, 8], [8, 5], [5, 3], [1, 10], [10, 4], [2, 10], [10, 5]],
        'tennis': [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [2, 5], [5, 6], [6, 0]],
    }
    points_colors = [
        [0.,    255.,  255.],
        [0.,   255.,    170.],
        [0., 170., 255.,],
        [85., 170., 255.],
        [0.,   255.,   85.], # 4
        [0., 85., 255.],
        [170., 85., 255.],
        [0.,   255.,   0.], # 7
        [0., 0., 255.], 
        [255., 0., 255.],
        [0.,    255.,  0.], # 10
        [0., 0., 255.],
        [255., 85., 170.],
        [170., 0, 255.],
        [255., 0., 170.],
        [255., 170., 85.],
        [85., 0., 255.],
        [255., 0., 85],
        [32., 0., 255.],
        [255., 0, 32],
        [0., 0., 255.],
        [255., 0., 0.],
    ]

    line_thickness = 2
    thickness = 4
    lineType = 8

    for bone in keypoint_linkages[obj_name]:
        idx1, idx2 = bone
        x1, y1 = keypoints_2d[idx1]
        x2, y2 = keypoints_2d[idx2]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), tuple(points_colors[idx1]), line_thickness, lineType)

    for i, points in enumerate(keypoints_2d):
        x, y = points
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), thickness, points_colors[i], thickness=-1, lineType=lineType)

    return image


def draw_boxes(image, person_box, object_box, hoi_box):
    image = cv2.rectangle(image, (int(hoi_box[0]), int(hoi_box[1])), (int(hoi_box[2]), int(hoi_box[3])), (255, 255, 255), 5)
    image = cv2.rectangle(image, (int(person_box[0]), int(person_box[1])), (int(person_box[2]), int(person_box[3])), (255, 0, 0), 2)
    image = cv2.rectangle(image, (int(object_box[0]), int(object_box[1])), (int(object_box[2]), int(object_box[3])), (0, 0, 255), 2)
    return image