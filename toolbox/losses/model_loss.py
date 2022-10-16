from torch import nn
import torch
from .losses import *

import torch.nn.functional as F


class SummaryLoss():
    def __init__(self):
        self.loss_summary_criteria = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

        self.loss_summary = np.zeros(shape=(16, 2))  # [sum_loss, num_pixel]

    def update(self, predict, target):
        loss = self.loss_summary_criteria(predict, target).view(-1)
        label_flatten = target.view(-1)

        for i in range(16):
            self.loss_summary[i][0] += loss[label_flatten == i].sum().item()
            self.loss_summary[i][1] += (label_flatten == i).sum()

    def reset(self):
        self.loss_summary = np.zeros(shape=(16, 2))  # [sum, num_pixel]

    @property
    def info(self):
        loss_ratio = self.loss_summary[:, 0] / (self.loss_summary[:, 0].sum() + 1e-7)
        loss_per_class_mean = self.loss_summary[:, 0] / (self.loss_summary[:, 1] + 1)
        info = '\n'
        for i in range(16):
            info += f'class_{i}: loss_rate={loss_ratio[i]:.4f} loss_mean={loss_per_class_mean[i]:.4f}\n'
        return info


class PLD_and_Seg_Loss(nn.Module):

    def __init__(self, n_classes=16):
        super(PLD_and_Seg_Loss, self).__init__()

        self.loss_cls_fn = MultiClassBCELoss(n_classes=n_classes)

        self.loss_summary = SummaryLoss()

    def forward(self, pred_dict, target, target_freespace):
        loss = self.loss_cls_fn(pred_dict['pred_cls'], target)

        self.loss_summary.update(pred_dict['pred_cls'], target)
        return {'loss': loss}


class BaseLineLoss(nn.Module):

    def __init__(self, cfg, score_thres=0.7, ignore_index=255):
        super(BaseLineLoss, self).__init__()
        # n_min = cfg.imgs_per_gpu * cfg.image_h * cfg.image_w // 16
        # self.cls_ohem = OhemCELossV2(thresh=score_thres, n_min=n_min, ignore_index=ignore_index)
        self.loss_cls_fn = MultiClassBCELoss(n_classes=16)

        self.loss_summary = SummaryLoss()

    def forward(self, pred_dict, target, target_freespace):
        loss = self.loss_cls_fn(pred_dict['pred_cls'], target)

        self.loss_summary.update(pred_dict['pred_cls'], target)
        return {'loss': loss}


class BaseLineLossV1(nn.Module):
    def __init__(self, cfg, score_thres=0.7, ignore_index=255):
        super(BaseLineLossV1, self).__init__()
        n_min = cfg.imgs_per_gpu * cfg.image_h * cfg.image_w // 16
        self.cls_ohem = OhemCELossV2(thresh=score_thres, n_min=n_min, ignore_index=ignore_index)
        self.cls_ce = MscCrossEntropyLoss()
        # self.loss_cls_fn = MultiClassBCELoss(n_classes=16)

        self.loss_summary = SummaryLoss()

    def forward(self, pred_dict, target, target_freespace):
        loss1 = self.cls_ohem(pred_dict['pred_cls'], target)
        loss2 = self.cls_ce(pred_dict['pred_cls_out8'], target)
        loss3 = self.cls_ce(pred_dict['pred_cls_out16'], target)

        self.loss_summary.update(pred_dict['pred_cls'], target)
        loss = loss1 + loss2 * 0.25 + loss3 * 0.125
        return {'loss': loss, 'loss1': loss1}


class BaseLineLossV2(nn.Module):

    def __init__(self, cfg, score_thres=0.7, ignore_index=255):
        super(BaseLineLossV2, self).__init__()
        n_min = cfg.imgs_per_gpu * cfg.image_h * cfg.image_w // 16
        self.cls_ohem = OhemCELossV2(thresh=score_thres, n_min=n_min, ignore_index=ignore_index)
        self.cls_Msc_ce = MscCrossEntropyLoss()
        # self.loss_cls_fn = MultiClassBCELoss(n_classes=16)

        self.loss_summary = SummaryLoss()

    def forward(self, pred_dict, target_dict):
        target = target_dict['label_cls']
        target_cls_encode = target_dict['label_cls_encode']

        # 4-scale loss for final output
        loss1 = self.cls_ohem(pred_dict['pred_cls'], target)

        # aux seg cls
        aux_list = (pred_dict['pred_cls_out8'], pred_dict['pred_cls_out16'], pred_dict['pred_cls_out32'])
        loss2 = self.cls_Msc_ce(aux_list, target)

        # aux global cls
        loss3 = F.binary_cross_entropy_with_logits(pred_dict['pred_g_cls'].squeeze(2).squeeze(2), target_cls_encode)

        self.loss_summary.update(pred_dict['pred_cls'], target)
        loss = loss1 + loss2 * 0.5 + loss3 * 0.25
        return {'loss': loss, 'loss1': loss1}


class BaseLineLossV3(nn.Module):

    def __init__(self, cfg, score_thres=0.7, ignore_index=255):
        super(BaseLineLossV3, self).__init__()
        n_min = cfg.imgs_per_gpu * cfg.image_h * cfg.image_w // 16
        self.cls_ohem = OhemCELossV2(thresh=score_thres, n_min=n_min, ignore_index=ignore_index)
        self.cls_Msc_ce = MscCrossEntropyLoss()
        # self.loss_cls_fn = MultiClassBCELoss(n_classes=16)
        self.detail_loss = BoundaryLoss()

        self.loss_summary = SummaryLoss()

    def forward(self, pred_dict, target_dict):
        target = target_dict['label_cls'].cuda()
        target_body = target_dict['label_body'].cuda()
        target_edge = target_dict['label_edge'].cuda()
        target_cls_encode = target_dict['label_cls_encode'].cuda()

        # print('shape:============')
        # print(target.shape, target_body.shape, target_edge.shape, target_cls_encode.shape)
        # print(pred_dict['pred_cls'].shape, pred_dict['pred_body'].shape, pred_dict['pred_edge'].shape, pred_dict['pred_g_cls'].shape)
        # print('==================')

        # loss for final output
        loss_final = self.cls_ohem(pred_dict['pred_cls'], target)

        # body Loss
        loss_body = self.cls_ohem(pred_dict['pred_body'], target_body)

        # edge Loss
        bce_loss, dice_loss = self.detail_loss(pred_dict['pred_edge'], target_edge.unsqueeze(1))
        loss_edge = bce_loss + dice_loss

        # aux seg cls
        aux_list = (pred_dict['pred_cls_out8'], pred_dict['pred_cls_out16'], pred_dict['pred_cls_out32'])
        loss_aux = self.cls_Msc_ce(aux_list, target)

        # aux global cls
        loss_g_cls = F.binary_cross_entropy_with_logits(pred_dict['pred_g_cls'].squeeze(2).squeeze(2), target_cls_encode)

        self.loss_summary.update(pred_dict['pred_cls'], target)
        loss = loss_final + loss_body + loss_edge + loss_aux * 0.5 + loss_g_cls * 0.5

        loss_dict = {
            'loss': loss,
            'loss_final': loss_final,
            'loss_body': loss_body,
            'loss_edge': loss_edge,
            'loss_aux': loss_aux * 0.5,
            'loss_g_cls': loss_g_cls * 0.5,
        }

        return loss_dict




class PLD_and_Seg_LossV2(nn.Module):

    def __init__(self, n_classes=16):
        super(PLD_and_Seg_LossV2, self).__init__()

        # 0, none
        # 1, car
        # 2, road
        # 3, ped
        # 4, zebra(nan)
        # 5, parkline(nan)
        # 6, arrow(nan)
        # 7, curbstone
        # 8, wallcolumn
        # 9, speed bump
        # 10, lane line(nan)
        # 11, park lock
        # 12, vehicle stoper
        # 13, traffic cone
        # 14, obstacle
        # 15, park area(nan)

        # 0 use in all obj
        self.large_obj_ids = [1, 2, 8, 14]
        self.small_obj_ids = [3, 7, 9, 11, 12, 13]

        self.loss_summary = SummaryLoss()

        self.n_classes = 16

    def forward(self, pred_dict, target, target_freespace):
        # {
        #     'pred_large_obj': pred_large_obj,
        #     'pred_small_obj': pred_small_obj,
        #     'pred_freespace': pred_freespace
        # }
        loss_dict = {}

        # freespace
        loss_freespace = F.binary_cross_entropy_with_logits(pred_dict['pred_freespace'].squeeze(1), target_freespace)
        loss_dict['loss_freespace'] = loss_freespace

        # cls
        target_flatten = target.view(-1)
        mask = target_flatten != 255

        pred_large_obj = pred_dict['pred_large_obj'].permute(0, 2, 3, 1).contiguous().view(-1, self.n_classes)
        pred_small_obj = pred_dict['pred_small_obj'].permute(0, 2, 3, 1).contiguous().view(-1, self.n_classes)

        target_flatten = target_flatten[mask]
        pred_large_obj = pred_large_obj[mask]
        pred_small_obj = pred_small_obj[mask]

        target_large_flatten = target_flatten.clone()
        target_large_flatten_one_hot = F.one_hot(target_large_flatten, self.n_classes).float()
        for obj_id in self.small_obj_ids:
            target_large_flatten_one_hot[:, obj_id] = 0

        target_small_flatten = target_flatten.clone()
        target_small_flatten_one_hot = F.one_hot(target_small_flatten, self.n_classes).float()
        for obj_id in self.large_obj_ids:
            target_small_flatten_one_hot[:, obj_id] = 0

        loss_large_obj = F.binary_cross_entropy_with_logits(input=pred_large_obj, target=target_large_flatten_one_hot)
        loss_small_obj = F.binary_cross_entropy_with_logits(input=pred_small_obj, target=target_small_flatten_one_hot)

        loss_dict['loss_large_obj'] = loss_large_obj
        loss_dict['loss_small_obj'] = loss_small_obj

        # self.loss_summary.update(pred_dict['pred_cls'], target)
        loss_dict['loss'] = sum([loss_dict[key] for key in loss_dict.keys() if key != 'loss'])
        return loss_dict


class PLD_and_Seg_LossV3(nn.Module):

    def __init__(self, n_classes=16):
        super(PLD_and_Seg_LossV3, self).__init__()

        # 0, none
        # 1, car
        # 2, road
        # 3, ped
        # 4, zebra(nan)
        # 5, parkline(nan)
        # 6, arrow(nan)
        # 7, curbstone
        # 8, wallcolumn
        # 9, speed bump
        # 10, lane line(nan)
        # 11, park lock
        # 12, vehicle stoper
        # 13, traffic cone
        # 14, obstacle
        # 15, park area(nan)

        # 0 use in all obj
        self.large_obj_ids = [1, 2, 8, 14]
        self.small_obj_ids = [3, 7, 9, 11, 12, 13]

        self.loss_summary = SummaryLoss()

        self.n_classes = 16

    def forward(self, pred_dict, target, target_freespace):
        # {
        #     'pred_large_obj': pred_large_obj,
        #     'pred_small_obj': pred_small_obj,
        #     'pred_freespace': pred_freespace
        # }
        loss_dict = {}

        # freespace
        loss_freespace = F.binary_cross_entropy_with_logits(pred_dict['pred_freespace'].squeeze(1), target_freespace)
        loss_dict['loss_freespace'] = loss_freespace

        # cls
        target_flatten = target.view(-1)
        mask = target_flatten != 255

        pred_cls = pred_dict['pred_cls'].permute(0, 2, 3, 1).contiguous().view(-1, self.n_classes)

        target_flatten = target_flatten[mask]
        pred_cls = pred_cls[mask]

        large_obj_mask = target_flatten < -1  # init all false
        for large_obj_id in self.large_obj_ids:
            large_obj_mask[target_flatten == large_obj_id] = True
        small_obj_mask = ~large_obj_mask

        target_flatten_one_hot = F.one_hot(target_flatten, self.n_classes).float()
        loss_large_obj = F.binary_cross_entropy_with_logits(input=pred_cls[large_obj_mask],
                                                            target=target_flatten_one_hot[large_obj_mask])
        loss_small_obj = F.binary_cross_entropy_with_logits(input=pred_cls[small_obj_mask],
                                                            target=target_flatten_one_hot[small_obj_mask])

        loss_dict['loss_large_obj'] = loss_large_obj
        loss_dict['loss_small_obj'] = loss_small_obj

        self.loss_summary.update(pred_dict['pred_cls'], target)
        loss_dict['loss'] = sum([loss_dict[key] for key in loss_dict.keys() if key != 'loss'])
        return loss_dict


class RepVGGLossV3(nn.Module):
    def __init__(self, cfg=None, score_thres=0.7, ignore_index=255):
        super(RepVGGLossV3, self).__init__()

        n_min = cfg.imgs_per_gpu * cfg.image_h * cfg.image_w // 16
        self.cls_ohem = OhemCELossV2(thresh=score_thres, n_min=n_min, ignore_index=ignore_index)
        self.detail_loss = DetailAggregateLossV2()

        self.loss_summary = [[0, 0] for i in range(16)]
        self.out_info = []

    def forward(self, pred_dict, target, target_freespace):
        loss_dict = {}

        # cls loss
        loss_cls, batch_loss_summary = self.cls_ohem(pred_dict['pred_cls'], target)
        loss_dict["loss_cls"] = loss_cls

        rst = []
        for i in range(16):
            self.loss_summary[i][0] += batch_loss_summary[i][0].item()
            self.loss_summary[i][1] += batch_loss_summary[i][1].item()
            rst.append((i, f'{self.loss_summary[i][0] / (self.loss_summary[i][1] + 1):.4f}'))
        self.out_info = rst

        # detail loss
        bce_loss, dice_loss = self.detail_loss(pred_dict, target)
        loss_dict["bound_bce"] = bce_loss
        loss_dict["bound_dice"] = dice_loss

        loss_dict['loss'] = sum([loss_dict[key] for key in loss_dict.keys() if key != 'loss'])
        return loss_dict


class RepVGGLossV4(nn.Module):
    def __init__(self, cfg=None, score_thres=0.7, ignore_index=255):
        super(RepVGGLossV4, self).__init__()

        n_min = cfg.imgs_per_gpu * cfg.image_h * cfg.image_w // 16
        self.cls_ohem = OhemCELossV2(thresh=score_thres, n_min=n_min, ignore_index=ignore_index)
        self.detail_loss = DetailAggregateLossV2()

        self.loss_summary = SummaryLoss()

    def forward(self, pred_dict, target, target_freespace):
        loss_dict = {}

        # freespace
        loss_freespace = F.binary_cross_entropy_with_logits(pred_dict['pred_freespace'].squeeze(1), target_freespace)
        loss_dict['loss_freespace'] = loss_freespace

        # cls loss
        loss_cls = self.cls_ohem(pred_dict['pred_cls'], target)
        loss_dict["loss_cls"] = loss_cls

        # detail loss
        bce_loss, dice_loss = self.detail_loss(pred_dict, target)
        loss_dict["bound_bce"] = bce_loss
        loss_dict["bound_dice"] = dice_loss

        loss_dict['loss'] = sum([loss_dict[key] for key in loss_dict.keys() if key != 'loss'])

        self.loss_summary.update(pred_dict['pred_cls'], target)
        return loss_dict
