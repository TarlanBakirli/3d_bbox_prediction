import torch
import os
import numpy as np

from skimage import io
from torch.utils.data import Dataset

from utils.utils import get_oriented_bboxes, filter_pointcloud_by_bboxes_vectorized, convert_pc_to_standard_format

class BBoxPredictionDataset(Dataset):
    def __init__(self, root_path, transform):
        self.root_path = root_path
        self.scenes = os.listdir(self.root_path)
        self.data_transforms = transform

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        scenes_path = os.path.join(self.root_path, self.scenes[idx])
        
        bbox_file = os.path.join(scenes_path, "bbox3d.npy")
        img_file = os.path.join(scenes_path, "rgb.jpg")
        pc_file = os.path.join(scenes_path, "pc.npy")
        mask_file = os.path.join(scenes_path, "mask.npy")

        
        bbox = np.load(bbox_file)
        image = io.imread(img_file)
        pc = np.load(pc_file)
        mask = np.load(mask_file)

        target = {'scene_id': self.scenes[idx], 'bbox': bbox}

        sample = {'image': image, 'pc': pc, 'mask': mask, 'target': target}

        if self.data_transforms is not None:
            sample = self.data_transforms(sample)

        bbox_2d_vertices, bbox_2d = get_oriented_bboxes(mask)
        filtered_pc = self.preprocess_point_cloud(pc, bbox_2d_vertices)

        bbox_2d = torch.from_numpy(bbox_2d)
        sample['bbox_2d'] = bbox_2d
        sample['pc'] = filtered_pc

        return sample['image'], sample['target'], sample['bbox_2d'], sample['pc']

    def preprocess_point_cloud(self, pc, bbox_2d):
        filtered_pc, pc_mask = filter_pointcloud_by_bboxes_vectorized(pc, bbox_2d)
        filtered_pc = convert_pc_to_standard_format(filtered_pointcloud=filtered_pc)
        return filtered_pc