from skimage.io import imsave, imread
from skimage.color import gray2rgb
from skimage.feature import canny
from  skimage.morphology import dilation, square


def visulize_gt(img, gt_img, mode='r'):
    """
    visualize mannual delination
    on the img
    :param img: 
    :param gt_img: 
    :return: 
    """
    rgb_img = gray2rgb(img)
    gt_edges = canny(gt_img, sigma=3)
    gt_edges = dilation(gt_edges, square(2))
    if mode=='r':
        rgb_img[gt_edges == 1, 0] = 255
        rgb_img[gt_edges == 1, 1] = 0
        rgb_img[gt_edges == 1, 2] = 0
    elif mode=='g':
        rgb_img[gt_edges == 1, 0] = 0
        rgb_img[gt_edges == 1, 1] = 255
        rgb_img[gt_edges == 1, 2] = 0
    elif mode=='b':
        rgb_img[gt_edges == 1, 0] = 0
        rgb_img[gt_edges == 1, 1] = 0
        rgb_img[gt_edges == 1, 2] = 255

    return rgb_img




