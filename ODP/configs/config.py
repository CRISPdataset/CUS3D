import os
import sys
import argparse
import yaml
from easydict import EasyDict

class Config():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_config', type=str, default='configs/default.yaml', help='path to config file')
        self.parser.add_argument("--maxpool", action="store_true", help="use max pooling to aggregate features (use majority voting in label projection mode)")
        self.parser.add_argument(
            "--config-file",
            default="configs/ovseg_swinB_vitL_demo.yaml",
            metavar="FILE",
            help="path to config file",
        )
        self.parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=['MODEL.WEIGHTS', '/home/wxj/code/P2P/cus3d/odp/pth/ovseg_swinbase_vitL14_ft_mpt.pth'],
            nargs=argparse.REMAINDER,
        )

    def get_config(self):
        cfgs = self.parser.parse_args()
        assert cfgs.data_config is not None
        with open(cfgs.data_config, 'r') as f:
            config = yaml.safe_load(f)
        for key in config:
            for k, v in config[key].items():
                setattr(cfgs, k, v)
        self.set_paths_cfg(cfgs)
        return cfgs

    def set_paths_cfg(self, CONF):
        CONF.PATH = EasyDict()
        CONF.PATH.BASE = CONF.root_path
        CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
        CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannetv2")
        CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
        CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
        CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

        # append to syspath
        for _, path in CONF.PATH.items():
            sys.path.append(path)

        # scannet data
        CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
        CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
        CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, CONF.scannet_data_folder)

        # scanref data
        CONF.SCANNET_FRAMES_ROOT = os.path.join(CONF.scannetv2_root, 'frames_square')
        CONF.PROJECTION = os.path.join(CONF.scannetv2_root, 'multiview_projection_scanrefer')
        
        CONF.CLIP_FEATURES_ROOT = os.path.join(CONF.scannetv2_root, '2dclip_features')
        CONF.CLIP_FEATURES_SUBROOT = os.path.join(CONF.CLIP_FEATURES_ROOT, "{}") # scene_id
        CONF.CLIP_FEATURES_PATH = os.path.join(CONF.CLIP_FEATURES_SUBROOT, "{}.npy") # frame_id
        
        CONF.SAM_FEATURES_ROOT = os.path.join(CONF.scannetv2_root, 'sam_features')
        CONF.SAM_FEATURES_SUBROOT = os.path.join(CONF.SAM_FEATURES_ROOT, "{}") # scene_id
        CONF.SAM_FEATURES_PATH = os.path.join(CONF.SAM_FEATURES_SUBROOT, "{}.npy") # frame_id
        
        CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode 
        # CONF.SCENE_NAMES = sorted(os.listdir(CONF.PATH.SCANNET_SCANS))
        
        #CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "avg_feats_color.hdf5")
        CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_maxpool.hdf5")# 实际上是avgpool
        CONF.MULTIVIEW_MAX = os.path.join(CONF.PATH.SCANNET_DATA, "clip_feats_maxpool.hdf5")
        CONF.MULTIVIEW_SAM = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_avgpool_sam.hdf5")#
        CONF.MULTIVIEW_200 = os.path.join(CONF.PATH.SCANNET_DATA, "scannet200_clip_features.hdf5")
        CONF.MULTIVIEW_TEST = os.path.join(CONF.PATH.SCANNET_DATA, "scannet_TEST.hdf5")
        
        #CONF.MULTIVIEW_MAX = os.path.join(CONF.PATH.SCANNET_DATA, "maxpooling_feats_color.hdf5")
        CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")

        # scannet split
        CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
        CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
        CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
        CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train2.txt")

        # output
        CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")
              
CONF = Config().get_config()
