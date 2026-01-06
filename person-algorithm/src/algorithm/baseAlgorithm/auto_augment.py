import cv2
import numpy as np

def Autolevelsadjust(img):
    # hists = []
    # for i in range(3):
    #     hist, x = np.histogram(img[:, :, i].ravel(), bins=256, range=(0, 256))
    #     hists.append(hist)
    # BHist, GHist, RHist = hists
    LowCut = 0.5
    HighCut = 100 - 0.5
    # max_percentile_pixel = np.percentile(BHist, 0.5*0.01)
    BMax, BMin = Compute(img[:, :, 0], LowCut, HighCut)
    GMax, GMin = Compute(img[:, :, 1], LowCut, HighCut)
    RMax, RMin = Compute(img[:, :, 2], LowCut, HighCut)
    img[:, :, 0] = Adjust(img[:, :, 0], BMin, BMax)
    img[:, :, 1] = Adjust(img[:, :, 1], GMin, GMax)
    img[:, :, 2] = Adjust(img[:, :, 2], RMin, RMax)
    return img

def Autocontrastadjust(img):
    LowCut = 0.5
    HighCut = 100 - 0.5
    # max_percentile_pixel = np.percentile(BHist, 0.5*0.01)
    BMax, BMin = Compute(img[:, :, 0], LowCut, HighCut)
    GMax, GMin = Compute(img[:, :, 1], LowCut, HighCut)
    RMax, RMin = Compute(img[:, :, 2], LowCut, HighCut)
    Max = max(max(BMax, GMax), RMax)
    Min = min(min(BMin, GMin), RMin)
    img = Adjust(img, Min, Max)
    return img

def Adjust(img, Min, Max):
    # image = img.copy()
    # img = np.where(img <= Min, 0, img)
    # img = np.where(img >= Max, 255, img)
    # img = np.where((img > Min) & (img < Max),
    #                ((img - Min) / (Max - Min) * 255).astype(np.uint8), img)
    cv2.normalize(img, img, Min, Max, cv2.NORM_MINMAX)
    return img

def Compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""


    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def Aug(src):
    """图像亮度增强"""
    if get_lightness(src) > 130:
        print("图片亮度足够，不做增强")
        return src
    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。

    max_percentile_pixel, min_percentile_pixel = Compute(src, 1, 99)

    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

    return out

def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()
    return lightness


def Autogammagraytrans(img):  # gamma函数处理
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gamma_val = np.log10(0.5) / np.log10(img_gray.mean() / 255)
    gamma_table = [np.power(x / 255.0, gamma_val) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.array(gamma_table).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)

def AutogammaHSVtrans(img):  # gamma函数处理
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    V = img_hsv[:, :, 2]
    PDF, x = np.histogram(V.ravel(), bins=256, range=(0, 256))
    PDF = PDF / (V.shape[0] * V.shape[1])
    gamma_table = [np.power(x / 255.0, 1 - PDF[:x].sum()) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.array(gamma_table).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)

