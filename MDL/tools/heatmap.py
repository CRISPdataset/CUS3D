import argparse
import multiprocessing as mp
import os
import os.path as osp

import numpy as np
import torch
import yaml
from munch import Munch
from nets.data import build_dataloader, build_dataset
from nets.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou)
from nets.model import NetsUnet
from nets.util import (collect_results_cpu, get_dist_info, get_root_logger, init_dist,
                            is_main_process, load_checkpoint, rle_decode)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser('cus3d')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--out', type=str, help='directory for output results')
    parser.add_argument('--text', nargs='+', help='directory for output results')
    args = parser.parse_args()
    return args


def save_npy(root, name, scan_ids, arrs):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.npy') for i in scan_ids]
    pool = mp.Pool()
    pool.starmap(np.save, zip(paths, arrs))
    pool.close()
    pool.join()


import sys
sys.path.append("../") # HACK add the root folder
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils.predictor import OVSegPredictor

from configs.odp.dict_config import CONF

SCANNET_LABELS_SOFT = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub']

# SCANNET_LABELS_SOFT = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair', 'table',
#                'bookcase', 'sofa', 'board']

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

def main():
    args = CONF
    cfg = setup_ovseg_cfg(args)
    predictor = OVSegPredictor(cfg)

    
    args = get_args()
    text_input = args.text
    text_features = predictor.get_text_features([' '.join(args.text)])
    text_features = text_features[:-1, :]
    del predictor
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.dist:
        init_dist()
    logger = get_root_logger()

    model = NetsUnet(**cfg.model).cuda()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f'Load state dict from {args.checkpoint}')
    load_checkpoint(args.checkpoint, logger, model)
    # print(cfg.data.test)
    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=args.dist, **cfg.dataloader.test)
    results = []
    scan_ids, coords, colors, prob_hard, prob_soft = [], [], [], [], []
    clip_fcs = []
    clip_preds = []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(dataloader) * world_size, disable=not is_main_process())
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            result = model(batch, text_features=text_features,demo='else')
            results.append(result)
            # clip_fcs.append(batch['clip_fcs'])
            progress_bar.update(world_size)
        progress_bar.close()
        
    if is_main_process():
        for res in results:
            scan_ids.append(res['scan_id'])
            coords.append(res['coords_float'])
            colors.append(res['color_feats'])
            prob_hard.append(res['prob_hard'])
            prob_soft.append(res['prob_soft'])
        for i, scan_id in enumerate(scan_ids):
            p1, c1 = vis(args, scan_id, coords[i], np.copy(colors[i]), prob_soft[i])
            # p2, c2 = vis2(args, scan_id, coords[i], np.copy(colors[i]))
            # p = np.concatenate((p1,p2),axis=0)
            # c = np.concatenate((c1,c2),axis=0)
            # vis3(args, scan_id, p, c)
            
from visualization import write_ply
import matplotlib as mpl

def float2color(c1,c2,mix=0,c0 = 'white'): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    if mix <= 0.5:
        mix *= 2
        # print(mix)
        c1=np.array(mpl.colors.to_rgb(c1))
        c2=np.array(mpl.colors.to_rgb(c0))
        # print(c1, c2)
        # print(mpl.colors.to_rgb((1-mix)*c1 + mix*c2))
        return mpl.colors.to_rgb((1-mix)*c1 + mix*c2)
    else:
        mix = (mix-0.5)*2
        mix = np.power(mix, 0.5)
        # if mix<0.5:
        #     mix = np.power(mix, 0.3)
        # else:
        #     mix = np.power(mix, 0.7)
        c1=np.array(mpl.colors.to_rgb(c0))
        c2=np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_rgb((1-mix)*c1 + (mix)*c2)


# def float2color(zero2one):
#     x = zero2one * 256 * 256 * 256
#     r = x % 256
#     g = ((x - r)/256 % 256)
#     b = ((x - r - g * 256)/(256*256) % 256)
#     r = round(float(r/256),2)
#     g = round(float(g/256),2)
#     b = round(float(b/256),2)
#     return [r,g,b]
# c1='#0047AB' #blue
c0 = 'white'
c1 = '#0047AB'
c2 = '#FF2400'

def vis(args, scan_id, points, colors, prob):
    # 根据prob制作colors
    # 随机扔掉一些数据
    # choices = np.random.choice(points.shape[0], points.shape[0]//10, replace=False)
    # points=points[choices]
    # prob=prob[choices]
    # colors=colors[choices]
    for i in range(prob.shape[0]):
        if np.isnan(prob[i]):
            colors[i, :3] = mpl.colors.to_rgb(c1)
        else:
            colors[i, :3] = float2color(c1, c2, prob[i])
        
    os.makedirs(os.path.join(args.out), exist_ok=True)
    outname = os.path.join(args.out, '-'.join(args.text) + '_' + scan_id + '.ply')
    write_ply(points, colors, None, outname)
    return points, colors
    
def vis2(args, scan_id, points, colors):
    os.makedirs(os.path.join(args.out), exist_ok=True)
    colors = (colors + 1)/2
    # points = points + 0.05
    outname = os.path.join(args.out, '-'.join(args.text) + '_' + scan_id + '_rgb.ply')
    write_ply(points, colors, None, outname)
    return points, colors

def vis3(args, scan_id, points, colors):
    os.makedirs(os.path.join(args.out), exist_ok=True)
    outname = os.path.join(args.out, 'COMBINED-'+ '-'.join(args.text) + '_' + scan_id + '.ply')
    write_ply(points, colors, None, outname)
    return points, colors

        
if __name__ == '__main__':
    main()
