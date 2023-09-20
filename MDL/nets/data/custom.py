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
ENET_FEATURE_DATABASE = '/home/wxj/code/P2P/ov-seg/data/scannet/scannet_data/enet_feats_maxpool.hdf5'
def to_tensor(arr):
    return torch.Tensor(arr).cuda()
class CustomDataset(Dataset):

    CLASSES = None
    NYU_ID = None

    def __init__(self,
                 data_root,
                 prefix,
                 suffix,
                 voxel_cfg=None,
                 training=True,
                 with_label=True,
                 repeat=1,
                 logger=None,
                 mask_labels=[],
                 scene_ids=[]):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.with_label = with_label
        self.repeat = repeat
        self.logger = logger
        self.mode = 'train' if training else 'test'
        self.scene_ids = scene_ids
        self.filenames = self.get_filenames()
        self.logger.info(f'Load {self.mode} dataset: {len(self.filenames)} scans')
        self.db = h5py.File(ENET_FEATURE_DATABASE, "r", libver="latest")
        self.remapper = np.ones(150) * (-100)
        for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
            self.remapper[x] = i
        self.mask_labels = []
        # self.mask_labels = [5, 12] #6sofa, 14desk
        # self.mask_labels = [5, 12, 9, 16]
        # self.mask_labels = [3,5,7,8,9,11,12,13,16,18]

    def get_filenames(self):
        ans = []
        if isinstance(self.prefix, str):
            filenames = glob(osp.join(self.data_root, self.prefix, '*' + self.suffix))
            assert len(filenames) > 0, 'Empty dataset.'
            filenames = sorted(filenames * self.repeat)
            return filenames
        for pfx in self.prefix:
            if len(self.scene_ids) > 0:
                for id in self.scene_ids:
                    filenames = glob(osp.join(self.data_root, pfx, id + '*' + self.suffix))
                    filenames = sorted(filenames * self.repeat)
                    ans += filenames
        assert len(ans) > 0, 'Empty dataset.'
        return ans

    def load(self, filename):
        return torch.load(filename)

    def __len__(self):
        return len(self.filenames)

    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [
            scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0)
            for n in noise
        ]
        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])
        return x + g(x) * mag
    
    def dataAugment(self, xyz, jitter=False, flip=False, rot=False, scale=False, prob=1.0):
        m = np.eye(3)
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.1
        if flip and np.random.rand() < prob:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        else:
            # Empirically, slightly rotate the scene can match the results from checkpoint
            theta = 0.35 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        if scale and np.random.rand() < prob:
            scale_factor = np.random.uniform(0.95, 1.05)
            xyz = xyz * scale_factor
        return np.matmul(xyz, m)

    def crop(self, xyz, step=32):
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]
        spatial_shape = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.voxel_cfg.max_npoint):
            step_temp = step
            if valid_idxs.sum() > 1e6:
                step_temp = step * 2
            offset = np.clip(spatial_shape - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < spatial_shape).sum(1) == 3)
            spatial_shape[:2] -= step_temp
        return xyz_offset, valid_idxs


    def transform_train(self, xyz, rgb, semantic_label, aug_prob=1.0):
        xyz_middle = self.dataAugment(xyz, True, True, True, aug_prob)
        xyz = xyz_middle * self.voxel_cfg.scale
        if np.random.rand() < aug_prob:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        # xyz_middle = xyz / self.voxel_cfg.scale
        xyz = xyz - xyz.min(0)
        max_tries = 5
        while (max_tries > 0):
            xyz_offset, valid_idxs = self.crop(xyz)
            if valid_idxs.sum() >= self.voxel_cfg.min_npoint:
                xyz = xyz_offset
                break
            max_tries -= 1
        if valid_idxs.sum() < self.voxel_cfg.min_npoint:
            return None
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        return xyz, xyz_middle, rgb, semantic_label

    def transform_test(self, xyz, rgb, semantic_label):
        xyz_middle = self.dataAugment(xyz, False, False, False, False)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        return xyz, xyz_middle, rgb, semantic_label
    
    def mask_unseen_points(self, xyz, rgb, sem, clip_fc):
        if len(self.mask_labels) == 0:
            return (xyz, rgb, sem), clip_fc
        idx_list = []
        for label in self.mask_labels:
            # sem = sem[sem != self.mask_labels[i]]
            idx = np.nonzero((sem != label))[0]
            sem = sem[idx]
            idx_list.append(idx)
        for idx in idx_list:
            rgb = rgb[idx]
            xyz = xyz[idx]
            clip_fc = clip_fc[idx]
        return (xyz, rgb, sem), clip_fc
    
    def reserve_unseen_points(self, xyz, rgb, sem, clip_fc):
        if len(self.mask_labels) == 0:
            return (xyz, rgb, sem), clip_fc
        idx_list = []
        for label in self.mask_labels:
            # sem = sem[sem != self.mask_labels[i]]
            idx = np.nonzero((sem == label))[0]
            idx_list.append(idx)
        idx = np.concatenate(idx_list)
        rgb = rgb[idx]
        xyz = xyz[idx]
        clip_fc = clip_fc[idx]
        return (xyz, rgb, sem), clip_fc
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, '')
        
        # print("start", filename)
        data = self.load(filename)
        # print("end", filename)
        #TODO: 进行5w的适配
        sem_data = np.load(f"/home/wxj/code/P2P/ov-seg/data/scannet/scannet_data/{scan_id}_sem_label.npy").astype(np.int32)
        sem_data = self.remapper[sem_data]
        
        # #data[0],data[1]
        # # #TODO: 输入完整的
        # with open(f"/home/wxj/code/P2P/ov-seg/data/scannet/scans/{scan_id}/{scan_id}_vh_clean_2.ply", 'rb') as f:
        #     import plyfile
        #     plydata = plyfile.PlyData.read(f)
        #     num_verts = plydata['vertex'].count
        #     vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        #     vertices[:,0] = plydata['vertex'].data['x']
        #     vertices[:,1] = plydata['vertex'].data['y']
        #     vertices[:,2] = plydata['vertex'].data['z']
        #     vertices[:,3] = plydata['vertex'].data['red']
        #     vertices[:,4] = plydata['vertex'].data['green']
        #     vertices[:,5] = plydata['vertex'].data['blue']

        # coords = np.ascontiguousarray(vertices[:, :3] - vertices[:, :3].mean(0))
        # colors = np.ascontiguousarray(vertices[:, 3:6]) / 127.5 - 1
        # data = (coords, colors, sem_data, ins_data)
        
        data = (data[0], data[1], sem_data)
    
        clip_fc = torch.Tensor(self.db[scan_id][()])
        # TODO: mask unseen labels
        data, clip_fc = self.mask_unseen_points(*data, clip_fc)
        # data, clip_fc = self.reserve_unseen_points(*data, clip_fc)
        
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        if data is None:
            return None
        xyz, xyz_middle, rgb, semantic_label = data
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb).float()
        if self.training:
            feat += torch.randn(feat.size(1)) * 0.1
        semantic_label = torch.from_numpy(semantic_label)
        
        
        # TMP:
        # clip_fc = torch.zeros((feat.shape))
        return (scan_id, coord, coord_float, feat, semantic_label, clip_fc)
        # scan_id, coord, coord_float, feat, semantic_label, clip_fc
        
    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        clip_fcs = []
        semantic_labels = []

        batch_id = 0

        for data in batch:
            if data is None:
                continue
            (scan_id, coord, coord_float, feat, semantic_label, clip_fc) = data
            scan_ids.append(scan_id)
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            clip_fcs.append(clip_fc)
            semantic_labels.append(semantic_label)
            batch_id += 1
            
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        clip_fcs = torch.cat(clip_fcs, 0)
        semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)
        spatial_shape = np.clip(
            coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
        return {
            'scan_ids': scan_ids,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'semantic_labels': semantic_labels,
            'spatial_shape': spatial_shape,
            'batch_size': batch_id,
            'clip_fcs': clip_fcs
        }
