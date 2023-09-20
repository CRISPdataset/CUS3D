# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import h5py
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement
from PIL import Image
import numpy as np

import sys
sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from configs.config import CONF
from lib.projection import ProjectionHelper
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from open_vocab_seg import add_ovseg_config
SCANNET_LIST = CONF.SCANNETV2_LIST
SCANNET_DATA = CONF.PATH.SCANNET_DATA
SCANNET_FRAME_ROOT = CONF.SCANNET_FRAMES
SCANNET_FRAME_PATH = os.path.join(SCANNET_FRAME_ROOT, "{}") # name of the file


ENET_FEATURE_DATABASE = CONF.MULTIVIEW # use avgpool
ENET_FEATURE_DATABASE_TEST = CONF.MULTIVIEW_TEST # use avgpool

# projection
INTRINSICS = [[37.01983, 0, 20, 0],[0, 38.52470, 15.5, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
PROJECTOR = ProjectionHelper(INTRINSICS, 0.1, 4.0, [41, 32], 0.05)

ENET_GT_PATH = SCANNET_FRAME_PATH

NYU40_LABELS = CONF.NYU40_LABELS
SCANNET_LABELS = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refrigerator', 'picture', 'cabinet', 'guitar']


PC_LABEL_ROOT = os.path.join(CONF.PATH.OUTPUT, "projections_no2dno3d")
PC_LABEL_PATH = os.path.join(PC_LABEL_ROOT, "{}.ply")
import pandas as pd
def get_nyu40_labels():
    labels = ["unannotated"]
    labels += pd.read_csv(NYU40_LABELS)["nyu40class"].tolist()
    
    return labels

def get_prediction_to_raw():
    labels = get_nyu40_labels()
    mapping = {i: label for i, label in enumerate(labels)}

    return mapping
import random
def to_tensor(arr):
    return torch.Tensor(arr).cuda()
def visualize(scene_id, coords, labels, class_names=SCANNET_LABELS):
    palette = create_color_palette()
    # if class_names != SCANNET_LABELS: palette = create_random_color_palette(class_names)
    if class_names != SCANNET_LABELS: palette = create_color_palette(lambda x: f"a photo of a {x}")
    nyu_to_scannet = get_nyu_to_scannet()
    vertex = []

    for i in range(coords.shape[0]):
        if class_names[labels[i].cpu().item()] == 'unannotated':
            continue
        if np.random.random() < 0:
            rd_color = random.choice(list(palette.values()))
            # print(rd_color)
            vertex.append(
            (
                coords[i][0],
                coords[i][1],
                coords[i][2],
                rd_color[0],
                rd_color[1],
                rd_color[2]
            )
        
        )    
        else:
            vertex.append(
                (
                    coords[i][0],
                    coords[i][1],
                    coords[i][2],
                    palette[class_names[labels[i].cpu().item()]][0],
                    palette[class_names[labels[i].cpu().item()]][1],
                    palette[class_names[labels[i].cpu().item()]][2]
                )
            )
    
    vertex = np.array(
        vertex,
        dtype=[
            ("x", np.dtype("float32")), 
            ("y", np.dtype("float32")), 
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8"))
        ]
    )

    output_pc = PlyElement.describe(vertex, "vertex")
    output_pc = PlyData([output_pc])
    os.makedirs(PC_LABEL_ROOT, exist_ok=True)
    output_pc.write(PC_LABEL_PATH.format(scene_id))

def get_nyu_to_scannet():
    nyu_idx_to_nyu_label = get_prediction_to_raw()
    scannet_label_to_scannet_idx = {label: i for i, label in enumerate(SCANNET_LABELS)}

    # mapping
    nyu_to_scannet = {}
    for nyu_idx in range(41):
        nyu_label = nyu_idx_to_nyu_label[nyu_idx]
        if nyu_label in scannet_label_to_scannet_idx.keys():
            scannet_idx = scannet_label_to_scannet_idx[nyu_label]
        else:
            scannet_idx = 0
        nyu_to_scannet[nyu_idx] = scannet_idx

    return nyu_to_scannet

def create_color_palette(func=None):
    res = {
        "unannotated": (0, 0, 0),
        "floor": (152, 223, 138),
        "wall": (174, 199, 232),
        "cabinet": (31, 119, 180),
        "bed": (255, 187, 120),
        "chair": (188, 189, 34),
        "sofa": (140, 86, 75),
        "table": (255, 152, 150),
        "door": (214, 39, 40),
        "window": (197, 176, 213),
        "bookshelf": (148, 103, 189),
        "picture": (196, 156, 148),
        "counter": (23, 190, 207),
        "desk": (247, 182, 210),
        "curtain": (219, 219, 141),
        "refrigerator": (255, 127, 14),
        "bathtub": (227, 119, 194),
        "shower curtain": (158, 218, 229),
        "toilet": (44, 160, 44),
        "sink": (112, 128, 144),
        "guitar": (82, 84, 163),
        "otherfurniture": (82, 84, 163),
    }
    if func is None:
        return res
    new_res = {}
    for k in res:
        new_res[func(k)] = res[k]
    return new_res
def create_random_color_palette(class_names):
    res = {}
    for name in class_names:
        res[name] = tuple(list(np.random.choice(range(256), size=3)))
    return res
    # return {
    #     "unannotated": (0, 0, 0),
    #     "floor": (152, 223, 138),
    #     "wall": (174, 199, 232),
    #     "cabinet": (31, 119, 180),
    #     "bed": (255, 187, 120),
    #     "chair": (188, 189, 34),
    #     "sofa": (140, 86, 75),
    #     "table": (255, 152, 150),
    #     "door": (214, 39, 40),
    #     "window": (197, 176, 213),
    #     "bookshelf": (148, 103, 189),
    #     "picture": (196, 156, 148),
    #     "counter": (23, 190, 207),
    #     "desk": (247, 182, 210),
    #     "curtain": (219, 219, 141),
    #     "refridgerator": (255, 127, 14),
    #     "bathtub": (227, 119, 194),
    #     "shower curtain": (158, 218, 229),
    #     "toilet": (44, 160, 44),
    #     "sink": (112, 128, 144),
    #     "otherfurniture": (82, 84, 163),
    # }
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_all_class_names():
    class_labels = '/home/wxj/code/P2P/ov-seg/resources/scannetv2-labels.combined.tsv'
    tsv_file = pd.read_csv(
        class_labels,
        sep='\t',
        header=0,
        index_col='id'
    ) 
    class_names = tsv_file['category'].tolist()
    return class_names

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/ovseg_swinB_vitL_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--class-lebels",
        help="class label file path",
        metavar="FILE",
        default='',
    )
    return parser
from open_vocab_seg.utils import VisualizationDemo
# 1. 读取feature
def run_one(scene_id='scene0000_00'):
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    cfg = setup_cfg(args)

    scene = np.load(os.path.join(SCANNET_DATA, scene_id)+"_vert.npy")[:, :3]

    model = VisualizationDemo(cfg)
    # class_names = args.class_names
    # args.class_labels = '/home/wxj/code/P2P/ov-seg/resources/scannetv2-labels.combined.tsv'
    # args.class_labels = None
    # if args.class_labels:
    #     tsv_file = pd.read_csv(
    #         args.class_labels,
    #         sep='\t',
    #         header=0,
    #         index_col='id'
    #     ) 
    #     class_names = tsv_file['category'].tolist()
    # TODO：class names 选择
    # class_names = get_nyu40_labels()
    class_names = SCANNET_LABELS
    # class_names = get_all_class_names()
    # class_names = [f"a photo of a {c}" for c in class_names]
    text_features = model.get_text_feature(class_names)
    
    with h5py.File(ENET_FEATURE_DATABASE_TEST, "r", libver="latest") as database:
        image_features = to_tensor(database[scene_id][()]) # 50000, 768
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True) # 99, 768
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # [240, 320, 4]
        values, indices = similarity[:, :].topk(1)
        visualize(scene_id, scene, indices, class_names)
        
def get_scene_train_list():
    return os.listdir('../dataset/scans/')
    
if __name__ == '__main__':
    # python tools/visualize_multiview_features.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names 'chair' 'table' 'curtain' 'Flooring' 'paper'  --input ./resources/demo_samples/0_0.jpg --output ./pred --opts MODEL.WEIGHTS pth/ovseg_swinbase_vitL14_ft_mpt.pth
    run_one("scene0000_00")
    # scenes = get_scene_train_list()
    # for scene in tqdm.tqdm(scenes):
    #     if os.path.exists(PC_LABEL_PATH.format(scene)):
    #         print('skip ' + scene)
    #         continue
    #     # if scene == 'scene0000_00':
    #     #     continue
    #     run_one(scene)
    
