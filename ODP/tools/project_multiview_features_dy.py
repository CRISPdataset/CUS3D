import os
import sys
import h5py
import torch
import torch.nn as nn
import shutil

import argparse
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import math
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import defaultdict
sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from configs.config import CONF
from lib.projection import ProjectionHelper
from open_vocab_seg.utils.predictor import OVSegPredictor
from detectron2.config import get_cfg
import numpy as np
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils import VisualizationDemo

import warnings
from data.scannetv2.conf import CLASS_LABELS_20
warnings.filterwarnings("ignore")
# SCANNET_LABELS = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refrigerator', 'picture', 'cabinet', 'otherfurniture']
SCANNET_LABELS = list(CLASS_LABELS_20)

SCANNET_LIST = CONF.SCANNETV2_LIST
SCANNET_TEST = CONF.SCANNETV2_TEST
SCANNET_TRAIN = CONF.SCANNETV2_TRAIN
SCANNET_DATA = CONF.PATH.SCANNET_DATA
SCANNET_FRAME_ROOT = CONF.SCANNET_FRAMES
SCANNET_FRAME_PATH = os.path.join(SCANNET_FRAME_ROOT, "{}") # name of the file

CLIP_FEATURE_PATH = CONF.CLIP_FEATURES_PATH
CLIP_FEATURE_DATABASE_TEST = CONF.MULTIVIEW_TEST

# projection
INTRINSICS = [[37.01983, 0, 20, 0],[0, 38.52470, 15.5, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
PROJECTOR = ProjectionHelper(INTRINSICS, 0.1, 4.0, [41, 32], 0.05)


CLIP_FEATURE_ROOT = CONF.CLIP_FEATURES_SUBROOT
CLIP_FEATURE_PATH = CONF.CLIP_FEATURES_PATH

def get_scene_list():
    with open(SCANNET_LIST, 'r') as f:
        return sorted(list(set(f.read().splitlines())))
def get_scene_test_list():
    with open(SCANNET_TEST, 'r') as f:
        return sorted(list(set(f.read().splitlines())))
def get_scene_train_list():
    with open(SCANNET_TRAIN, 'r') as f:
        return sorted(list(set(f.read().splitlines())))

def to_tensor(arr):
    return torch.Tensor(arr).cuda()

def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    
    return image

def load_image(file, image_dims):
    image = imread(file)
    # preprocess
    image = resize_crop_image(image, image_dims)
    if len(image.shape) == 3: # color image
        image =  np.transpose(image, [2, 0, 1])  # move feature to front
        image = transforms.Normalize(mean=[0.496342, 0.466664, 0.440796], std=[0.277856, 0.28623, 0.291129])(torch.Tensor(image.astype(np.float32) / 255.0))
    elif len(image.shape) == 2: # label image
#         image = np.expand_dims(image, 0)
        pass
    else:
        raise
        
    return image

def load_pose(filename):
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]

    return np.asarray(lines).astype(np.float32)

def load_depth(file, image_dims):
    depth_image = imread(file)
    # preprocess
    depth_image = resize_crop_image(depth_image, image_dims)
    depth_image = depth_image.astype(np.float32) / 1000.0

    return depth_image

def get_scene_data(scene_list):
    scene_data = {}
    scene_label = {}
    for scene_id in tqdm(scene_list):
        # load the original vertices, not the axis-aligned ones
        scene_data[scene_id] = np.load(os.path.join(SCANNET_DATA, scene_id)+"_vert.npy")[:, :3]
        scene_label[scene_id] = np.load(os.path.join(SCANNET_DATA, scene_id)+"_ins_label.npy")
    return scene_data, scene_label

def compute_projection(points, depth, camera_to_world):
    """
        :param points: tensor containing all points of the point cloud (num_points, 3)
        :param depth: depth map (size: proj_image)
        :param camera_to_world: camera pose (4, 4)
        
        :return indices_3d (array with point indices that correspond to a pixel),
        :return indices_2d (array with pixel indices that correspond to a point)

        note:
            the first digit of indices represents the number of relevant points
            the rest digits are for the projection mapping
    """
    num_points = points.shape[0]
    num_frames = depth.shape[0]
    indices_3ds = torch.zeros(num_frames, num_points + 1).long().cuda()
    indices_2ds = torch.zeros(num_frames, num_points + 1).long().cuda()

    for i in range(num_frames):
        indices = PROJECTOR.compute_projection(to_tensor(points), to_tensor(depth[i]), to_tensor(camera_to_world[i]))
        if indices:
            indices_3ds[i] = indices[0].long()
            indices_2ds[i] = indices[1].long()
            # print("found {} mappings in {} points from frame {}".format(indices_3ds[i][0], num_points, i))
        
    return indices_3ds, indices_2ds

def setup_ovseg_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_label_from_fc(image_features, text_features):
    image_features = image_features.permute(1, 2, 0)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True) # 99, 768
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[:, :].topk(1) # 240, 320, 1
    return indices

def get_label_from_fc_3d(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True) # 99, 768
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[:, :].topk(1) # 240, 320, 1
    return indices
import time
def load_frame_image(file):    
    image = torch.Tensor(read_image(file, format="BGR").astype(np.float32))
    return image
if __name__ == "__main__":
    args = CONF
    cfg = setup_ovseg_cfg(args)
    predictor = OVSegPredictor(cfg)
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    scene_list = get_scene_list()
    scene_data, scene_label = get_scene_data(scene_list)
    
    with h5py.File(CLIP_FEATURE_DATABASE_TEST, "w", libver="latest") as database:
        print("projecting multiview features to point cloud...")
        text_fc = predictor.get_text_features(SCANNET_LABELS).cuda()
        for scene_id in scene_list[:1]:
            start_time = time.time()
            print("processing {}...".format(scene_id))
            print('*'*30)
            print("start compute frame features...")

            frame_list = sorted(os.listdir(SCANNET_FRAME_ROOT.format(scene_id, "color")), key=lambda x:int(x.split(".")[0]))
            for frame_file in tqdm(frame_list):
                frame_id = int(frame_file.split(".")[0])
                image = load_frame_image(SCANNET_FRAME_PATH.format(scene_id, "color", "{}.jpg".format(frame_id)))
                image = image.numpy()
                # prediction, visualized_output, image_feature, text_feature = tclip_model.run_on_image(image, default_classes)
                image_feature = predictor(image, SCANNET_LABELS)['odp_fc']
                os.makedirs(CLIP_FEATURE_ROOT.format(scene_id), exist_ok=True)
                np.save(CLIP_FEATURE_PATH.format(scene_id, frame_id), image_feature.cpu().numpy())
                
            print("end compute frame features...")
            scene = scene_data[scene_id]
            label = to_tensor(scene_label[scene_id].astype(np.int32))
            # load frames
            frame_list = list(map(lambda x: x.split(".")[0], sorted(os.listdir(SCANNET_FRAME_ROOT.format(scene_id, "color")))))
            frame_list = frame_list[:]#TODO
            scene_images = np.zeros((len(frame_list), 3, 240, 320))
            scene_depths = np.zeros((len(frame_list), 32, 41))
            scene_poses = np.zeros((len(frame_list), 4, 4))
            for i, frame_id in tqdm(enumerate(frame_list)):
                img = read_image(SCANNET_FRAME_PATH.format(scene_id, "color", "{}.jpg".format(frame_id)), format="BGR")
                scene_depths[i] = load_depth(SCANNET_FRAME_PATH.format(scene_id, "depth", "{}.png".format(frame_id)), [41, 32])
                scene_poses[i] = load_pose(SCANNET_FRAME_PATH.format(scene_id, "pose", "{}.txt".format(frame_id)))

            # compute projections for each chunk
            projection_3d, projection_2d = compute_projection(scene, scene_depths, scene_poses)
            
            # compute valid projections
            projections = []
            for i in range(projection_3d.shape[0]):
                num_valid = projection_3d[i, 0]
                if num_valid == 0:
                    continue

                projections.append((frame_list[i], projection_3d[i], projection_2d[i], i))

            print(f"start project, total {len(projections)} projections")
            point_features = to_tensor(scene).new(scene.shape[0], 768).fill_(0)
            ins_features = to_tensor(scene).new(scene.shape[0], 768).fill_(0)
            ins_dict = defaultdict(lambda: defaultdict(list))
            clip_label_dict = defaultdict(lambda: defaultdict(list))
            for i, projection in tqdm(enumerate(projections)):
                frame_id = projection[0]
                projection_3d = projection[1]
                projection_2d = projection[2]
                proj_i = projection[3]
                feat = to_tensor(np.load(CLIP_FEATURE_PATH.format(scene_id, frame_id)))
                
                
                feat = F.interpolate(feat.unsqueeze(0), [32, 41]).squeeze(0)
                feat_labels = get_label_from_fc(feat, text_fc)
                feat_labels = feat_labels.permute(2, 0, 1).float()
                # feat_labels = F.interpolate(feat_labels.unsqueeze(0), [32, 41]).squeeze(0)

                proj_feat = PROJECTOR.project(feat, projection_3d, projection_2d, scene.shape[0]).transpose(1, 0)
                proj_label = PROJECTOR.project(feat_labels, projection_3d, projection_2d, scene.shape[0]).transpose(1, 0)

                if args.maxpool:
                    # # only apply max pooling on the overlapping points
                    # # find out the points that are covered in projection
                    feat_mask = ((proj_feat == 0).sum(1) != 768).bool()

                    label_mask = label[feat_mask].int().tolist() # gt
                    fcs = proj_feat[feat_mask].cpu() #投影的fc
                    fcs_label = proj_label[feat_mask].cpu() # 投影的label
                    feat_mask_sq = (feat_mask==True).nonzero().cpu() #对应的index
                    for t in range(fcs.shape[0]):
                        #1. 添加feat的index和fc
                        idx = feat_mask_sq[t].item()
                        ins_dict[label_mask[t]][idx].append(fcs[t])
                        #2. 添加feat的label
                        clip_label_dict[label_mask[t]][idx].append(fcs_label[t].item())
                        
                else:
                    if i == 0:
                        point_features = proj_feat
                    else:
                        mask = (point_features == 0).sum(1) == 768
                        point_features[mask] = proj_feat[mask]
            # 计算clip-label-dict统计频率
            # print(ins_dict.keys())
            for gt_id, features in ins_dict.items():
                # print(f"gt id@{gt_id}, {len(features)} points.")
                # features的key是点云的idx，value是[fc1, fc2, ...])
                labels = list(clip_label_dict[gt_id].values())
                counts = np.bincount(sum(labels, []))
                max_label = np.argmax(counts)
                # print(f"max label@{max_label}, {counts[max_label]} points.")
                # print(counts, end='\n\n')
                ins_fcs = []
                empty_ids = []
                test_fc = None
                for k, vs in clip_label_dict[gt_id].items():
                    np_vs = np.array(vs)
                    most_ids = np.where(np_vs == max_label)[0]
                    if most_ids.shape[0] == 0:
                        empty_ids.append(k)
                    else:    
                        local_fcs = torch.stack(features[k])
                        target_fcs = local_fcs[most_ids]
                        target_fcs = torch.mean(target_fcs, dim=0)
                        # target_fcs = torch.max(target_fcs, dim=0)[0]
                        # target_fcs = target_fcs[0]
                        ins_fcs.append(target_fcs)
                        point_features[k] = target_fcs
                #针对有空白点的部分，进行平均值填充
                ins_mean_fc = torch.mean(torch.stack(ins_fcs), dim=0)
                empty_ids = torch.tensor(empty_ids).long()
                point_features[empty_ids] = ins_mean_fc.cuda()
            
            database.create_dataset(scene_id, data=point_features.cpu().numpy())
            end_time = time.time()
            ###############刪除特征文件###############
            shutil.rmtree(CLIP_FEATURE_ROOT.format(scene_id))
            print(f"finish@{scene_id}, cost@{end_time - start_time}s")
    print("done!")

    
