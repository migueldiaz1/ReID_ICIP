# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .quadruplet_loss import QuadrupletLoss


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    metric_type = cfg.MODEL.METRIC_LOSS_TYPE
    use_label_smooth = cfg.MODEL.IF_LABELSMOOTH == 'on'

    triplet = None
    quadruplet = None
    xent = None

    if metric_type == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)
    elif metric_type == 'quadruplet':
        quadruplet = QuadrupletLoss(
            margin1=cfg.SOLVER.QUADRUPLET_MARGIN1,
            margin2=cfg.SOLVER.QUADRUPLET_MARGIN2
        ).to(cfg.MODEL.DEVICE)
    else:
        raise ValueError(f"Unsupported METRIC_LOSS_TYPE: {metric_type}")

    if use_label_smooth:
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    # define loss_func
    def loss_func(score, feat, target):
        ce_loss = xent(score, target) if use_label_smooth else F.cross_entropy(score, target)
        
        if sampler == 'softmax':
            return ce_loss

        elif sampler == 'triplet':
            if metric_type == 'triplet':
                return triplet(feat, target)[0]
            elif metric_type == 'quadruplet':
                return quadruplet(feat, target)

        elif sampler == 'softmax_triplet':
            if metric_type == 'triplet':
                return ce_loss + triplet(feat, target)[0]
            elif metric_type == 'quadruplet':
                return ce_loss + quadruplet(feat, target)

        raise ValueError(f"Unsupported sampler: {sampler} or METRIC_LOSS_TYPE: {metric_type}")

    # devolver también la instancia del criterio para que pueda moverse a GPU en el trainer
    return loss_func, triplet or quadruplet


def make_loss_with_center(cfg, num_classes):
    if cfg.MODEL.NAME in ['resnet18', 'resnet34']:
        feat_dim = 512
    else:
        feat_dim = 2048

    metric_type = cfg.MODEL.METRIC_LOSS_TYPE
    use_label_smooth = cfg.MODEL.IF_LABELSMOOTH == 'on'

    if metric_type == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
    elif metric_type == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
    else:
        raise ValueError(f"Unsupported METRIC_LOSS_TYPE with center: {metric_type}")

    if use_label_smooth:
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target):
        ce_loss = xent(score, target) if use_label_smooth else F.cross_entropy(score, target)
        center_loss = cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        if metric_type == 'center':
            return ce_loss + center_loss
        elif metric_type == 'triplet_center':
            return ce_loss + triplet(feat, target)[0] + center_loss

    return loss_func, center_criterion
