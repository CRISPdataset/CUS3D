import argparse
import multiprocessing as mp
import os
import os.path as osp

import numpy as np
import torch
import yaml
from munch import Munch
import sys
sys.path.append('./')
from nets.data import build_dataloader, build_dataset
from nets.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou,evaluate_semantic_iou_unseen)
from nets.model import NetsUnet, NetsUnetCls
from nets.util import (collect_results_cpu, get_dist_info, get_root_logger, init_dist,
                            is_main_process, load_checkpoint, rle_decode)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser('Cus3d')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--out', type=str, help='directory for output results')
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


def save_single_instance(root, scan_id, insts, nyu_id=None):
    f = open(osp.join(root, f'{scan_id}.txt'), 'w')
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    for i, inst in enumerate(insts):
        assert scan_id == inst['scan_id']
        label_id = inst['label_id']
        # scannet dataset use nyu_id for evaluation
        if nyu_id is not None:
            label_id = nyu_id[label_id - 1]
        conf = inst['conf']
        f.write(f'predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n')
        mask_path = osp.join(root, 'predicted_masks', f'{scan_id}_{i:03d}.txt')
        mask = rle_decode(inst['pred_mask'])
        np.savetxt(mask_path, mask, fmt='%d')
    f.close()


def save_pred_instances(root, name, scan_ids, pred_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    nyu_ids = [nyu_id] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, nyu_ids))
    pool.close()
    pool.join()


def save_gt_instance(path, gt_inst, nyu_id=None):
    if nyu_id is not None:
        sem = gt_inst // 1000
        ignore = sem == 0
        ins = gt_inst % 1000
        nyu_id = np.array(nyu_id)
        sem = nyu_id[sem - 1]
        sem[ignore] = 0
        gt_inst = sem * 1000 + ins
    np.savetxt(path, gt_inst, fmt='%d')


def save_gt_instances(root, name, scan_ids, gt_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.txt') for i in scan_ids]
    pool = mp.Pool()
    nyu_ids = [nyu_id] * len(scan_ids)
    pool.starmap(save_gt_instance, zip(paths, gt_insts, nyu_ids))
    pool.close()
    pool.join()


def save_panoptic_single(path, panoptic_pred, learning_map_inv, num_classes):
    # convert cls to kitti format
    panoptic_ids = panoptic_pred >> 16
    panoptic_cls = panoptic_pred & 0xFFFF
    new_learning_map_inv = {num_classes: 0}
    for k, v in learning_map_inv.items():
        if k == 0:
            continue
        if k < 9:
            new_k = k + 10
        else:
            new_k = k - 9
        new_learning_map_inv[new_k] = v
    panoptic_cls = np.vectorize(new_learning_map_inv.__getitem__)(panoptic_cls).astype(
        panoptic_pred.dtype)
    panoptic_pred = (panoptic_cls & 0xFFFF) | (panoptic_ids << 16)
    panoptic_pred.tofile(path)


def save_panoptic(root, name, scan_ids, arrs, learning_map_inv, num_classes):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.label'.replace('velodyne', 'predictions')) for i in scan_ids]
    learning_map_invs = [learning_map_inv] * len(scan_ids)
    num_classes_list = [num_classes] * len(scan_ids)
    for p in paths:
        os.makedirs(osp.dirname(p), exist_ok=True)
    pool = mp.Pool()
    pool.starmap(save_panoptic_single, zip(paths, arrs, learning_map_invs, num_classes_list))
import sys
# sys.path.append("../") # HACK add the root folder
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
sys.path.append("./")
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils.predictor import OVSegPredictor

from configs.odp.dict_config import CONF

SCANNET_LABELS_SOFT = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub']

# SCANNET_LABELS_SOFT += ['guitar', 'carpet', 'garbage can', 'seat']
# SCANNET_LABELS_SOFT += ['computer', 'dustbin', 'printer', 'television', 'pillow', 'footcloth']
# SCANNET_LABELS_SOFT += ['whiteboard', 'package bag', 'stereo']
# SCANNET_LABELS_SOFT += ['running machine', 'ball']

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
    text_features = predictor.get_text_features(SCANNET_LABELS_SOFT)
    text_features = text_features[:-1, :]
    del predictor
    
    args = get_args()
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
    scan_ids, coords, colors, sem_preds, sem_labels = [], [], [], [], []
    clip_preds = []
    offset_preds, offset_labels, inst_labels, pred_insts, gt_insts = [], [], [], [], []
    panoptic_preds = []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(dataloader) * world_size, disable=not is_main_process())
    
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            result = model(batch,text_features=text_features)
            # if i == 5:
            #     break
            results.append(result)
            progress_bar.update(world_size)
        progress_bar.close()
        results = collect_results_cpu(results, len(dataset))
    if is_main_process():
        for res in results:
            scan_ids.append(res['scan_id'])
            sem_labels.append(res['semantic_labels'])
            coords.append(res['coords_float'])
            colors.append(res['color_feats'])
            sem_preds.append(res['semantic_preds'])
            clip_preds.append(res['clip_preds'])
            print(clip_preds)


        logger.info('Evaluate semantic segmentation and offset MAE')
        ignore_label = cfg.model.ignore_label
        evaluate_semantic_miou(sem_preds, sem_labels, ignore_label, logger)
        evaluate_semantic_acc(sem_preds, sem_labels, ignore_label, logger)
        evaluate_semantic_iou_unseen(sem_preds, sem_labels, ignore_label, logger, [3,5,7,8,9,11,12,13,16,18]) # test for unseen exp
        evaluate_semantic_miou(clip_preds, sem_labels, ignore_label, logger)
        evaluate_semantic_acc(clip_preds, sem_labels, ignore_label, logger)
        evaluate_semantic_miou(sem_preds, clip_preds, ignore_label, logger)
        evaluate_semantic_acc(sem_preds, clip_preds, ignore_label, logger)
        # save output
        if not args.out:
            return
        logger.info('Save results')
        save_npy(args.out, 'coords', scan_ids, coords)
        save_npy(args.out, 'colors', scan_ids, colors)
        save_npy(args.out, 'semantic_pred', scan_ids, sem_preds)
        save_npy(args.out, 'clip_pred', scan_ids, clip_preds)
        save_npy(args.out, 'semantic_label', scan_ids, sem_labels)
        
if __name__ == '__main__':
    main()
