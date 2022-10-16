import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from toolbox.losses.lovasz_losses import lovasz_softmax
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import NLLLoss2d


class MultiClassBCELoss(nn.Module):
    def __init__(self, n_classes, ignore_index=255):
        super(MultiClassBCELoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        b, c, h, w = predict.shape

        predict = predict.permute(0, 2, 3, 1).contiguous().view(-1, c)
        target = target.view(-1)
        # mask = (target != self.ignore_index)
        mask = (target < self.n_classes) & (target != self.ignore_index)
        predict = predict[mask]
        target = target[mask]
        target_one_hot = F.one_hot(target, self.n_classes).float()
        loss = F.binary_cross_entropy_with_logits(input=predict, target=target_one_hot)
        return loss


class MscCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(MscCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        for item in input:
            h, w = item.size(2), item.size(3)
            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            loss += F.cross_entropy(item, item_target.squeeze(1).long(), weight=self.weight,
                                    ignore_index=self.ignore_index, reduction=self.reduction)
        return loss / len(input)


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_index=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class OhemBCELoss(nn.Module):
    def __init__(self, thresh, n_min=0.02):
        super(OhemBCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.criteria = nn.BCEWithLogitsLoss(reduce='none')

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


class OhemCELossV2(nn.Module):
    def __init__(self, thresh, n_min, ignore_index=255):
        super(OhemCELossV2, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)

        label_flatten = labels.view(-1)
        mask_wall_and_obstacle = (label_flatten == 8) | (label_flatten == 14)

        loss[mask_wall_and_obstacle] *= 1.0

        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class DetailAggregateLossV2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLossV2, self).__init__()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)

        self.fuse_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]], dtype=torch.float32).reshape(1, 3, 1, 1).type(
                torch.cuda.FloatTensor))

    def forward(self, pred_dict, gtmasks):
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        # boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
        #                                        dim=1)

        # boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        # boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid = boundary_targets_x2_up * 0.6 + boundary_targets_x4_up * 0.3 + boundary_targets_x8_up * 0.1

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        bce_loss_x4 = F.binary_cross_entropy_with_logits(pred_dict['pred_bound_x4'], boudary_targets_pyramid)
        dice_loss_x4 = dice_loss_func(torch.sigmoid(pred_dict['pred_bound_x4']), boudary_targets_pyramid)
        bce_loss_x8 = F.binary_cross_entropy_with_logits(pred_dict['pred_bound_x4'], boudary_targets_pyramid)
        dice_loss_x8 = dice_loss_func(torch.sigmoid(pred_dict['pred_bound_x4']), boudary_targets_pyramid)

        bce_loss = bce_loss_x4 * 0.75 + bce_loss_x8 * 0.25
        dice_loss = dice_loss_x4 * 0.75 + dice_loss_x8 * 0.25
        return bce_loss, dice_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            nowd_params += list(module.parameters())
        return nowd_params


class BoundaryLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BoundaryLoss, self).__init__()
        pass

    def forward(self, pred_boundary, gt_boundary):
        bce_loss = F.binary_cross_entropy_with_logits(pred_boundary, gt_boundary)
        dice_loss = dice_loss_func(torch.sigmoid(pred_boundary), gt_boundary)

        return bce_loss, dice_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            nowd_params += list(module.parameters())
        return nowd_params