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

class NetsUnet(nn.Module):
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
        # self.input_conv = spconv.SparseSequential(
        #     spconv.SubMConv3d(
        #     in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'), nn.ReLU()
        # )
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
        self.to_high_mlp = nn.Sequential(
            MLP(channels, 256, norm_fn, 2), nn.ReLU(),
            MLP(256, 768, norm_fn, 2))
 
        self.init_weights()
        for mod in fixed_modules:
            mod = getattr(self, mod)
            for name, param in mod.named_parameters():
                param.requires_grad = False
                if 'deconv' in name:
                    param.requires_grad = True
                    # print("un fix name: ", name)

        self.mask_ids = []
        self.mask_labels = [3,5,9, 12,16,18]

        
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

    def forward(self, batch, return_loss=False, text_features=None, scannet200=False,demo=False):
        if return_loss:
            return self.forward_train(**batch, text_features=text_features, scannet200=scannet200)
        elif demo:
            return self.forward_sim(**batch, text_features=text_features,demo=demo)    
        else:
            return self.forward_val(**batch, text_features=text_features)
    
    
    def forward_sim(self, voxel_coords, p2v_map, v2p_map, coords_float, feats, semantic_labels, spatial_shape, batch_size, scan_ids, clip_fcs, **kwargs):
        text_features = kwargs['text_features']
        mode = kwargs['demo']
        if mode == 'clip':
            if self.with_coords:
                feats = torch.cat((feats, coords_float), 1)
            voxel_feats = voxelization(feats, p2v_map)        
            input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
            output_feats = self.forward_backbone(input, v2p_map, x4_split=self.test_cfg.x4_split)
            output_feats = self.to_high_mlp(output_feats)
            #TODO: use clip fcs
            clip_fcs = torch.where(torch.isnan(clip_fcs), torch.full_like(clip_fcs, 1e-5), clip_fcs)
            clip_fcs = torch.where(torch.abs(clip_fcs) <1e-6, torch.full_like(clip_fcs, 1e-5), clip_fcs)
            idx = (torch.abs(clip_fcs) > 1e-5).nonzero().T.squeeze() #IDX
            idx = torch.unique(idx[0])
            clip_fcs = clip_fcs[idx]
            prob_hard, prob_soft, _ = self.get_cos_labels(clip_fcs, text_features, T1=100, T2=10) # 5
            prob_hard = prob_hard.squeeze()
            prob_soft = prob_soft.squeeze()
            from torch.nn.functional import normalize
            prob_soft = prob_soft.softmax(dim=0)
            prob_soft = (prob_soft - prob_soft.min()) / (prob_soft.max() - prob_soft.min())
  
            return dict(
                scan_id=scan_ids[0],
                coords_float=coords_float[idx].cpu().numpy(),
                color_feats=feats[idx].cpu().numpy(),
                semantic_labels=semantic_labels[idx].cpu().numpy(),
                prob_hard=prob_hard.cpu().numpy(),
                prob_soft=prob_soft.cpu().numpy()
            )
        else:
            if self.with_coords:
                feats = torch.cat((feats, coords_float), 1)
            voxel_feats = voxelization(feats, p2v_map)        
            input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
            output_feats = self.forward_backbone(input, v2p_map, x4_split=self.test_cfg.x4_split)
            output_feats = self.to_high_mlp(output_feats)
            #TODO: use clip fcs
            prob_hard, prob_soft, _ = self.get_cos_labels(output_feats, text_features, T1=100, T2=3) # 5
            prob_hard = prob_hard.squeeze()
            prob_soft = prob_soft.squeeze()
            from torch.nn.functional import normalize
            prob_soft = prob_soft.softmax(dim=0)
            prob_soft = (prob_soft - prob_soft.min()) / (prob_soft.max() - prob_soft.min())
            
            return dict(
                scan_id=scan_ids[0],
                coords_float=coords_float.cpu().numpy(),
                color_feats=feats.cpu().numpy(),
                semantic_labels=semantic_labels.cpu().numpy(),
                prob_hard=prob_hard.cpu().numpy(),
                prob_soft=prob_soft.cpu().numpy()
            )
    def get_cos_labels(self, image_features, text_features, T1=100, T2=100):
        """"default hard pred, T=1/100"""
        a = image_features / image_features.norm(dim=-1, keepdim=True)
        b = text_features / text_features.norm(dim=-1, keepdim=True) # 99, 768
        similarity_hard = (T1 * a @ b.T).softmax(dim=-1)
        indices = torch.argmax(similarity_hard, dim=1)
        similarity_soft = (T2 * a @ b.T)
        return similarity_hard, similarity_soft, indices
    
    @cuda_cast
    def forward_val(self, voxel_coords, p2v_map, v2p_map, coords_float, feats, semantic_labels, spatial_shape, batch_size, scan_ids, clip_fcs, **kwargs):
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)        
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        output_feats = self.forward_backbone(input, v2p_map, x4_split=self.test_cfg.x4_split)
        if self.test_cfg.x4_split:
            coords_float = self.merge_4_parts(coords_float)
            semantic_labels = self.merge_4_parts(semantic_labels)
        output_feats = self.to_high_mlp(output_feats)
        
        #TODO: 验证
        text_features = kwargs["text_features"]
        # print(output_feats.shape)
        unet_pred_cls_hard, unet_pred_cls_soft, cos_labels = self.get_cos_labels(output_feats, text_features)
        _, clip_pred_cls_soft, clip_labels = self.get_cos_labels(clip_fcs, text_features)
        return dict(
            scan_id=scan_ids[0],
            coords_float=coords_float.cpu().numpy(),
            color_feats=feats.cpu().numpy(),
            semantic_preds=cos_labels.cpu().numpy(),
            clip_preds=clip_labels.cpu().numpy(),
            semantic_labels=semantic_labels.cpu().numpy()
            )

    @cuda_cast
    def forward_train(self, voxel_coords, p2v_map, v2p_map, coords_float, feats, semantic_labels, spatial_shape, batch_size, scan_ids, clip_fcs, **kwargs):
        self.mask_ids = []            
        losses = {}
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)        
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        output_feats = self.forward_backbone(input, v2p_map)
        output_feats = self.to_high_mlp(output_feats)        
        text_features = kwargs["text_features"]
        scannet200 = kwargs["scannet200"]
        _, clip_pred_cls_soft, clip_labels = self.get_cos_labels(clip_fcs, text_features)
        # 对 hi_output_feats 添加约束
        valid_index = (clip_labels != 0).nonzero()
        ignore_indexes = (clip_labels == 0).nonzero()
                
        # clip lossess
        clip_loss = self.feature_loss(clip_fcs, output_feats, valid_index=valid_index)
        losses.update(clip_loss)
        #TODO: 处理text_features
        # text losses
        unet_pred_cls_hard, unet_pred_cls_soft, cos_labels = self.get_cos_labels(output_feats, text_features)
        clip_labels[ignore_indexes] = self.ignore_label
        # filter unvalid point
        unet_pred_cls_soft = unet_pred_cls_soft[valid_index].squeeze()
        clip_pred_cls_soft = clip_pred_cls_soft[valid_index].squeeze()

        if scannet200 is False:
            label_loss = self.label_loss(unet_pred_cls_hard, unet_pred_cls_soft, clip_labels, clip_pred_cls_soft)
        else:
            # use gt
            label_loss = self.label_loss(unet_pred_cls_hard, unet_pred_cls_soft, semantic_labels, None)
        losses.update(label_loss)

        if scannet200 is False:
            acc_loss = self.acc_loss(cos_labels, clip_labels, valid_index)
        else:
            acc_loss = self.acc_loss(cos_labels, semantic_labels, valid_index)
        losses.update(acc_loss)
    
        return self.parse_losses(losses)
    
    def style_loss(self, target_feature, input):
        target = gram_matrix(target_feature).detach()
        G = gram_matrix(input)
        loss = F.mse_loss(G, target)
        return loss
    
    def acc_loss(self, dt, gt, valid_idx):
        losses = {}
        valid_dt = dt[valid_idx].squeeze()
        valid_gt = gt[valid_idx].squeeze()
        match_num = (valid_dt == valid_gt).sum()
        losses['acc'] = match_num / valid_gt.shape[0] * 100
        return losses
    
    def feature_loss(self, clip_features, model_features, valid_index):
        losses = {}
        clip_features = clip_features[valid_index].squeeze()
        model_features = model_features[valid_index].squeeze()
        a = clip_features / clip_features.norm(dim=-1, keepdim=True)
        b = model_features / model_features.norm(dim=-1, keepdim=True)
 
        # model_features = F.normalize(model_features, p=2, dim=1)
        D = a * b
        # print("D", (D == torch.nan).nonzero())
        loss = 1 - torch.sum(D, dim=1)
        losses['clip_cos_loss'] = 30 * loss.mean()
        # mse loss
        
        mse_loss = F.mse_loss(clip_features, model_features, reduction='sum')
        mse_loss = mse_loss / clip_features.shape[0]
        losses['clip_mse'] = 10 * mse_loss
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

    def forward_backbone(self, input, input_map, x4_split=False):
        if x4_split:
            output_feats = self.forward_4_parts(input, input_map)
            output_feats = self.merge_4_parts(output_feats)
        else:
            output = self.input_conv(input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features
            output_feats = output_feats[input_map.long()]
        # semantic_scores = self.semantic_linear(output_feats)
        # pt_offsets = self.offset_linear(output_feats)
        return output_feats
    def forward_4_parts(self, x, input_map):
        """Helper function for s3dis: devide and forward 4 parts of a scene."""
        outs = []
        for i in range(4):
            inds = x.indices[:, 0] == i
            feats = x.features[inds]
            coords = x.indices[inds]
            coords[:, 0] = 0
            x_new = spconv.SparseConvTensor(
                indices=coords, features=feats, spatial_shape=x.spatial_shape, batch_size=1)
            out = self.input_conv(x_new)
            out = self.unet(out)
            out = self.output_layer(out)
            outs.append(out.features)
        outs = torch.cat(outs, dim=0)
        return outs[input_map.long()]
    def merge_4_parts(self, x):
        """Helper function for s3dis: take output of 4 parts and merge them."""
        inds = torch.arange(x.size(0), device=x.device)
        p1 = inds[::4]
        p2 = inds[1::4]
        p3 = inds[2::4]
        p4 = inds[3::4]
        ps = [p1, p2, p3, p4]
        x_split = torch.split(x, [p.size(0) for p in ps])
        x_new = torch.zeros_like(x)
        for i, p in enumerate(ps):
            x_new[p] = x_split[i]
        return x_new
