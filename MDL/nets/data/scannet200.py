import numpy as np
import torch

from .custom import CustomDataset

import math
import os.path as osp
from glob import glob
import h5py
import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch
from torch.utils.data import Dataset

from ..ops import voxelization_idx
class ScanNet200Dataset(CustomDataset):

    def load(self, filename):
        if self.with_label:
            return torch.load(filename)
        else:
            xyz, rgb = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, dummy_sem_label, dummy_inst_label

    # def getInstanceInfo(self, xyz, instance_label, semantic_label):
    #     ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
    #     instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
    #     # instance_cls = [x - 2 if x != -100 else x for x in instance_cls]
    #     #TODO: 进行5w的适配
    #     # instance_cls = [x - 1 if x != -100 else (x if x != -1 else -100) for x in instance_cls]
    #     return instance_num, instance_pointnum, instance_cls, pt_offset_label
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, '')
        
        # print("start", filename)
        data = self.load(filename)
        # print("end", filename)
        #TODO: 进行5w的适配
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        if data is None:
            return None
        xyz, xyz_middle, rgb, semantic_label, instance_label = data

        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb).float()
        if self.training:
            feat += torch.randn(feat.size(1)) * 0.1
        semantic_label = torch.from_numpy(semantic_label)
        # 拿到clip_fc
        clip_fc = torch.Tensor(self.db[scan_id][()])
        # TMP:
        # clip_fc = torch.zeros((feat.shape))
        return (scan_id, coord, coord_float, feat, semantic_label, clip_fc)
        # scan_id, coord, coord_float, feat, semantic_label, clip_fc