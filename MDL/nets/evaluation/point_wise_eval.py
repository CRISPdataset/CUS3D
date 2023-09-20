import numpy as np
# remapper = [-1 for _ in range(100)]
# for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
#     remapper[x] = i
SCANNET_LABELS_SOFT = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

# SCANNET_LABELS_SOFT = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair', 'table',
#                'bookcase', 'sofa', 'board']
# SCANNET_LABELS_SOFT = ['wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
# 'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
# 'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
# 'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
# 'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
# 'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
# 'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
# 'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
# 'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
# 'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
# 'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress']
def evaluate_semantic_acc(pred_list, gt_list, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    assert gt.shape == pred.shape
    correct = (gt[gt != ignore_label] == pred[gt != ignore_label]).sum()
    whole = (gt != ignore_label).sum()
    acc = correct.astype(float) / whole * 100
    logger.info(f'Acc: {acc:.1f}')
    return acc

def evaluate_semantic_iou_unseen(pred_list, gt_list, ignore_label=-100, logger=None, unseen_ids=[]):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    pos_inds = gt != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    assert gt.shape == pred.shape
    iou_list = []
    iou_list2 = []
    valid_cls_list = []
    valid_cls_list2 = []
    
    for _index in np.unique(gt):
        if _index != ignore_label and _index != 19:
            if _index not in unseen_ids: #seen
                intersection = ((gt == _index) & (pred == _index)).sum()
                union = ((gt == _index) | (pred == _index)).sum()
                iou = intersection.astype(float) / union * 100
                iou_list.append(iou)
                valid_cls_list.append(_index)
            else: #unseen
                intersection = ((gt == _index) & (pred == _index)).sum()
                union = ((gt == _index) | (pred == _index)).sum()
                iou = intersection.astype(float) / union * 100
                iou_list2.append(iou)
                valid_cls_list2.append(_index)

    miou = np.mean(iou_list)
    logger.info('Class-wise mIoU: ' + ' '.join(f'{x:.1f}({i}, {SCANNET_LABELS_SOFT[i]})' for i,x in zip(valid_cls_list, iou_list)))
    logger.info(f'mIoU(seen): {miou:.1f}')
    
    miou2 = np.mean(iou_list2)
    logger.info('Class-wise mIoU: ' + ' '.join(f'{x:.1f}({i}, {SCANNET_LABELS_SOFT[i]})' for i,x in zip(valid_cls_list2, iou_list2)))
    logger.info(f'mIoU(unseen): {miou2:.1f}')
    
    hiou = 2*miou * miou2 / (miou + miou2)
    logger.info(f'hIoU(2*a*b/(a+b)): {hiou:.1f}')
    return miou,miou2,hiou

def evaluate_semantic_miou(pred_list, gt_list, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    pos_inds = gt != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    assert gt.shape == pred.shape
    iou_list = []
    valid_cls_list = []
    
    for _index in np.unique(gt):
        if _index != ignore_label and _index != -1: #12: s3dis #19: scannetv2
            intersection = ((gt == _index) & (pred == _index)).sum()
            union = ((gt == _index) | (pred == _index)).sum()
            iou = intersection.astype(float) / union * 100
            iou_list.append(iou)
            valid_cls_list.append(_index)

    valid_iou_list = iou_list
    miou = np.mean(valid_iou_list)
    logger.info('Class-wise mIoU: ' + ' '.join(f'{x:.1f}({i}, {SCANNET_LABELS_SOFT[i]})' for i,x in zip(valid_cls_list, valid_iou_list)))
    logger.info(f'mIoU: {miou:.1f}')
    return miou


def evaluate_offset_mae(pred_list, gt_list, gt_instance_list, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    gt_instance = np.concatenate(gt_instance_list, axis=0)
    pos_inds = gt_instance != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    mae = np.abs(gt - pred).sum() / pos_inds.sum()
    logger.info(f'Offset MAE: {mae:.3f}')
    return mae
