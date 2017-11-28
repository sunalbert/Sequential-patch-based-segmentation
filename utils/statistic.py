import numpy as np
from skimage.measure import label, regionprops


def get_coordinate(mask):
    areas = regionprops(label(mask//255))
    assert len(areas) == 1
    centroid = np.rint(areas[0].centroid)
    return centroid.astype(np.int)


def dice_ratio(y_pred, y_true):
    """
    dice
    :param y_pred: fg: 1 bg: 0
    :param y_true: fg: 1 bg: 0
    :return:
    """
    smooth = 1.0
    y_pred_flatten = y_pred.flatten()
    y_true_flatten = y_true.flatten()
    intersection = y_pred_flatten * y_true_flatten
    dice = (2 * np.sum(intersection) + smooth) / (np.sum(y_true_flatten) + np.sum(y_pred_flatten) + smooth)
    print(dice)
    return dice


def cd_xy(mask, gt):
    """
    compute the cd_xy according to the
    mask and ground truth
    :param mask:
    :param gt:
    :return:
    """
    indcies = np.argwhere(mask == 255)
    center = np.rint(np.mean(indcies, axis=0))
    gt_center = get_coordinate(gt)
    return gt_center - center