import argparse
import datetime
import os
import os.path as osp
import shutil
import time
import sys
sys.path.append("../") # HACK add the root folder
import torch
import yaml
from munch import Munch
from nets.data import build_dataloader, build_dataset
from nets.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou)

from nets.model import NetsUnetCls, NetsUnet
from nets.util import (AverageMeter, SummaryWriter, build_optimizer, checkpoint_save,
                            collect_results_cpu, cosine_lr_after_step, get_dist_info,
                            get_max_memory, get_root_logger, init_dist, is_main_process,
                            is_multiple, is_power2, load_checkpoint, build_inner_mlp_optimizer)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
sys.path.append("./")
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils.predictor import OVSegPredictor

from configs.odp.dict_config import CONF

SCANNET_LABELS_SOFT = ['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

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

def get_args():
    parser = argparse.ArgumentParser('cus3d')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--skip_validate', action='store_true', help='skip validation')
    args = parser.parse_args()
    return args


def train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer, text_features=None):
    torch.autograd.set_detect_anomaly(True)  

    model.train()
    iter_time = AverageMeter(True)
    data_time = AverageMeter(True)
    meter_dict = {}
    end = time.time()

    if train_loader.sampler is not None and cfg.dist:
        train_loader.sampler.set_epoch(epoch)
    for i, batch in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)
        cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            loss, log_vars = model(batch, return_loss=True, text_features=text_features)
        # meter_dict
        for k, v in log_vars.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)    
        
        scaler.scale(loss).backward()

        # backward
        if (i + 1) % 4 == 0:
            if cfg.get('clip_grad_norm', None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        # time and print
        remain_iter = len(train_loader) * (cfg.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]['lr']

        if is_multiple(i, 10):
            log_str = f'Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  '
            log_str += f'lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, '\
                f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
            for k, v in meter_dict.items():
                log_str += f', {k}: {v.val:.4f} ({v.avg:.4f})'
            logger.info(log_str)
    writer.add_scalar('train/learning_rate', lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f'train/{k}', v.avg, epoch)
    checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq)
#520 72.7462 71.7462
def validate(epoch, model, val_loader, cfg, logger, writer, text_features=None):
    logger.info('Validation')
    results = []
    all_sem_preds, all_sem_labels= [], []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(val_loader) * world_size, disable=not is_main_process())

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            result = model(batch, text_features=text_features)
            results.append(result)
            progress_bar.update(world_size)
        progress_bar.close()
        # results = collect_results_cpu(results, len(val_set))
    if is_main_process():
        for res in results:
            all_sem_preds.append(res['semantic_preds'])
            all_sem_labels.append(res['semantic_labels'])

        logger.info('Evaluate semantic segmentation and offset MAE')
        miou = evaluate_semantic_miou(all_sem_preds, all_sem_labels, cfg.model.ignore_label,
                                    logger)
        acc = evaluate_semantic_acc(all_sem_preds, all_sem_labels, cfg.model.ignore_label,
                                    logger)

        writer.add_scalar('val/mIoU', miou, epoch)
        writer.add_scalar('val/Acc', acc, epoch)


def main():
    #####################ovseg
    args = CONF
    cfg = setup_ovseg_cfg(args)
    predictor = OVSegPredictor(cfg)
    text_features = predictor.get_text_features(SCANNET_LABELS_SOFT)
    text_features = text_features[:-1, :]
    del predictor
    ################################
    
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    if args.dist:
        init_dist()
    cfg.dist = args.dist

    # work_dir & logger
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{cfg_txt}')
    logger.info(f'Distributed: {args.dist}')
    logger.info(f'Mix precision training: {cfg.fp16}')
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # model
    model = NetsUnetCls(**cfg.model).cuda()
    
    
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], find_unused_parameters = False)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # data
    train_set = build_dataset(cfg.data.train, logger)
    val_set = build_dataset(cfg.data.test, logger)
    train_loader = build_dataloader(
        train_set, training=True, dist=args.dist, **cfg.dataloader.train)
    val_loader = build_dataloader(val_set, training=False, dist=args.dist, **cfg.dataloader.test)

    # TODO: 冻结optim
    optimizer = build_optimizer(model, cfg.optimizer)
    # optimizer = build_inner_mlp_optimizer(model, cfg.optimizer)

    # pretrain, resume
    # args.resume = False #no resume
    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif cfg.pretrain:
        logger.info(f'Load pretrain from {cfg.pretrain}')
        load_checkpoint(cfg.pretrain, logger, model)

    # train and val
    logger.info('Training')

    for epoch in range(start_epoch, cfg.epochs + 1):
        train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer, text_features)
        validate(epoch, model, val_loader, cfg, logger, writer, text_features)
        writer.flush()


if __name__ == '__main__':
    main()
