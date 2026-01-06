"""
Create by Chengqi.Lv
2020/5/7
"""


def dice_loss(input, target):
    smooth = 1.
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / ((iflat * iflat).sum() + (tflat * tflat).sum() + smooth))