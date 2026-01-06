"""
Create by Chengqi.Lv
2020/4/14
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


def _neg_loss(pred, gt):
    """
    Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class CenterNetFocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super(CenterNetFocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def _quality_focal_loss(pred, gt):
    """
        QFL from paper https://arxiv.org/pdf/2006.04388.pdf
    """
    beta = 2
    pos_inds = gt.gt(0.4).float()  # 大于0的都算正样本？？？
    num_pos = pos_inds.sum()  # TODO:研究一下正样本怎么选择

    loss = -torch.pow(torch.abs(gt - pred), beta) * ((1 - gt) * torch.log(1 - pred) + gt * torch.log(pred))
    loss = loss.sum() / max(num_pos, 1)
    return loss


class CenterNetQualityFocalLoss(nn.Module):
    """
        QFL from paper https://arxiv.org/pdf/2006.04388.pdf
    """
    def __init__(self):
        super(CenterNetQualityFocalLoss, self).__init__()
        self.neg_loss = _quality_focal_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)
