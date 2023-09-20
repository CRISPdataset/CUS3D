import functools
from collections import OrderedDict

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..ops import (ball_query, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from ..util import cuda_cast, force_fp32, rle_decode, rle_encode
from .blocks import MLP, ResidualBlock, UBlock

def gram_matrix(features):
    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(features.shape[0] * features.shape[1])

class NetsUnetCls(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels=32,
                 num_blocks=7,

                 semantic_classes=20,
                 semantic_weight=None,
                 ignore_label=-100,
                 with_coords=True,
                 train_cfg=None,
                 test_cfg=None,
                 fixed_modules=[]):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_classes = semantic_classes
        self.semantic_weight = semantic_weight
        self.ignore_label = ignore_label
        self.with_coords = with_coords
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # backbone
        if with_coords:
            in_channels += 3
            self.in_channels += 3
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
        self.to_high_mlp = nn.Sequential(
            MLP(channels, 256, norm_fn, 2), nn.ReLU(),
            MLP(256, 768, norm_fn, 2))

        self.cls_mlp = MLP(768, 20, norm_fn, 2)

        self.init_weights()
        for mod in fixed_modules:
            mod = getattr(self, mod)
            for name, param in mod.named_parameters():
                param.requires_grad = False
                if 'deconv' in name:
                    param.requires_grad = True
                    # print("un fix name: ", name)

        self.mask_ids = []
        self.mask_labels = [3,5,7,8,9,11,12,13,16,18]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, return_loss=False, text_features=None, scannet200=False):
        if return_loss:
            return self.forward_train(**batch, text_features=text_features, scannet200=scannet200)
        else:
            return self.forward_val(**batch,text_features=text_features)

    def get_cos_labels(self, image_features, text_features, T1=100, T2=100):
        """"default hard pred, T=1/100"""
        a = image_features / image_features.norm(dim=-1, keepdim=True)
        b = text_features / text_features.norm(dim=-1, keepdim=True) # 99, 768
        similarity_hard = (T1 * a @ b.T).softmax(dim=-1)
        indices = torch.argmax(similarity_hard, dim=1)
        similarity_soft = (T2 * a @ b.T)
        return similarity_hard, similarity_soft, indices
    @cuda_cast
    def forward_val(self, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                     semantic_labels, spatial_shape, batch_size,
                     scan_ids, clip_fcs, **kwargs):
        losses = {}
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)        
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        output_feats = self.forward_backbone(input, v2p_map)
        output_feats = self.to_high_mlp(output_feats)
        pred_cls = self.cls_mlp(output_feats)
        text_features = kwargs["text_features"]
        # print(output_feats.shape)
        unet_pred_cls_hard, unet_pred_cls_soft, cos_labels = self.get_cos_labels(output_feats, text_features)
        return dict(
            scan_id=scan_ids[0],
            coords_float=coords_float.cpu().numpy(),
            color_feats=feats.cpu().numpy(),
            semantic_preds=cos_labels.cpu().numpy(),
            semantic_labels=semantic_labels.cpu().numpy()
            )

    @cuda_cast
    def forward_train(self, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                      semantic_labels, spatial_shape, batch_size, clip_fcs, **kwargs):
        self.mask_ids = []            
        losses = {}
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)        
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        output_feats = self.forward_backbone(input, v2p_map)
        output_feats = self.to_high_mlp(output_feats)
        
        pred_cls = self.cls_mlp(output_feats)
        # DICE loss
        dice_loss = self.dice_loss(pred_cls, semantic_labels)
        losses.update(dice_loss)
        
        
        text_features = kwargs["text_features"]
        _, _, clip_labels = self.get_cos_labels(clip_fcs, text_features)
        # # 对 hi_output_feats 添加约束
        valid_index = (clip_labels != 0).nonzero()
        # Text emb gt
        text_gt_loss = self.text_cos_loss(output_feats, semantic_labels, text_features, valid_index)
        losses.update(text_gt_loss)
        
        # CLIP
        clip_loss = self.feature_loss(clip_fcs, output_feats, valid_index=valid_index)
        losses.update(clip_loss)
        # CLS
        cls_loss = self.cls_loss(pred_cls, semantic_labels)
        losses.update(cls_loss)
        # ACC：放进验证集？
        pred_cls_labels = torch.argmax(pred_cls, dim=1)
        acc_loss = self.acc_loss(pred_cls_labels, semantic_labels)
        losses.update(acc_loss)
        # MIOU
        miou_loss = self.miou_loss(pred_cls_labels, semantic_labels)
        losses.update(miou_loss)
        return self.parse_losses(losses)
    
    def dice_loss(self, dt, gt):
        valid_idx = (gt != self.ignore_label).nonzero()
        dt = dt.softmax(dim=1)
        gt = gt[valid_idx].squeeze()
        dt = dt[valid_idx].squeeze()
        targets_one_hot = torch.nn.functional.one_hot(gt, num_classes=dt.shape[1])
        intersection = torch.sum(targets_one_hot * dt, dim=0)
        mod_b = torch.sum(targets_one_hot, dim=0)
        dice_coefficient = (1. + 2. * intersection) / (intersection.square() + mod_b.square() + 1.)
        dice_loss = 1-dice_coefficient

        loss = dice_loss.mean()
        losses = {}
        losses['dice'] = loss
        return losses
    
    def text_cos_loss(self, image_features, gt, text_features, valid_index):
        gt = gt[valid_index].squeeze()
        image_features = image_features[valid_index].squeeze()
        valid_index = (gt != self.ignore_label).nonzero()
        gt = gt[valid_index].squeeze()
        image_features = image_features[valid_index].squeeze()
        
        text_features = text_features[1:] # 20 * 768
        
        gt_text_features = F.one_hot(gt, num_classes=text_features.shape[0]).to(torch.float) # N * 20
        gt_text_features = gt_text_features @ text_features # N * 768
        # cos loss
        loss = 1 - torch.cosine_similarity(image_features, gt_text_features, dim=-1)
        losses = {}
        losses['text_cos_loss'] = loss.mean() #*30!!!
        return losses
    
    def style_loss(self, target_feature, input):
        target = gram_matrix(target_feature).detach()
        G = gram_matrix(input)
        loss = F.mse_loss(G, target)
        return loss
    
    def miou_loss(self, pred, gt):
        losses = {}
        pos_inds = gt != self.ignore_label
        gt = gt[pos_inds]
        pred = pred[pos_inds]
        assert gt.shape == pred.shape
        iou_list = []
        for _index in torch.unique(gt):
            if _index != self.ignore_label:
                intersection = ((gt == _index) & (pred == _index)).sum()
                union = ((gt == _index) | (pred == _index)).sum()
                iou = intersection.to(torch.float) / union * 100
                iou_list.append(iou)
        
        miou = torch.mean(torch.tensor(iou_list).cuda())
        
        losses["miou_gt"] = miou
        return losses
    
    def acc_loss(self, dt, gt):
        losses = {}
        correct = (gt[gt != self.ignore_label] == dt[gt != self.ignore_label]).sum()
        whole = (gt != self.ignore_label).sum()
        acc = correct.to(torch.float32) / whole * 100
        losses['acc_gt'] = acc
        return losses
    
    def feature_loss(self, clip_features, model_features, valid_index):
        losses = {}
        clip_features = clip_features[valid_index].squeeze()
        model_features = model_features[valid_index].squeeze()
        a = clip_features / clip_features.norm(dim=-1, keepdim=True)
        b = model_features / model_features.norm(dim=-1, keepdim=True)
        D = a * b
        loss = 1 - torch.sum(D, dim=1)
        losses['clip_cos'] = loss.mean() #*30!!!
        # mse loss
        
        mse_loss = F.mse_loss(clip_features, model_features, reduction='sum')
        mse_loss = mse_loss / clip_features.shape[0]
        losses['clip_mse'] = mse_loss # *10!!!
        return losses
    
    def cls_loss(self, pred_cls, sem_labels):
        losses = {}
        if self.semantic_weight:
            weight = torch.tensor(self.semantic_weight, dtype=torch.float, device='cuda')
        else:
            weight = None
        loss = F.cross_entropy(pred_cls, sem_labels, weight=weight, ignore_index=self.ignore_label)
        losses['cls_loss'] = loss
        return losses
        
    def label_loss(self, semantic_scores, semantic_scores_soft, semantic_labels, semantic_labels_soft):
        losses = {}
        if self.semantic_weight:
            weight = torch.tensor(self.semantic_weight, dtype=torch.float, device='cuda')
        else:
            weight = None

        label_loss = F.cross_entropy(
            semantic_scores, semantic_labels, weight=weight, ignore_index=self.ignore_label)
        losses['hard_label'] = label_loss
        
        # soft loss
        if semantic_labels_soft is not None:
            semantic_labels_soft = F.softmax(semantic_labels_soft, dim=1)
            # data = torch.sum((F.log_softmax(semantic_scores_soft, dim=1) * semantic_labels_soft), dim=1)

            soft_loss = F.cross_entropy(semantic_scores_soft, semantic_labels_soft, weight=weight, ignore_index=self.ignore_label)
            losses['soft_label'] = soft_loss
        return losses

    def parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' + f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        # log_vars['loss_wo_text'] = loss - log_vars['label_loss']
        
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
    def forward_backbone(self, input, input_map):
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features    
        output_feats = output_feats[input_map.long()]
        return output_feats