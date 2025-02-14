import torch
from pytorch3d.ops import box3d_overlap


def eval_iou3d(bbox_pred, bbox_gt):

    intersection_vol, iou_3d = box3d_overlap(bbox_pred, bbox_gt)
    iou = torch.diagonal(iou_3d)

    return iou