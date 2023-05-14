from typing import Dict
import numpy as np
import torch
import cv2
import pickle
import json


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, Dict):
            batch[k] = to_device(v, device)
    return batch


def extract_bbox_from_mask(mask):
    indices = np.array(np.nonzero(np.array(mask)))

    try:
        y1 = np.min(indices[0, :])
        y2 = np.max(indices[0, :])
        x1 = np.min(indices[1, :])
        x2 = np.max(indices[1, :])

        return np.array([x1, y1, x2, y2])
    except:
        return np.zeros(4)


def load_mask(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def read_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except:
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='iso-8859-1')

    return data


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
