from skimage.io import imsave, imread
import os
from utils.visualize import visulize_gt
from glob import glob
import numpy as np
from skimage.measure import regionprops, label


def get_coordinate(mask):
    areas = regionprops(label(mask//255))
    assert len(areas) == 1
    centroid = np.rint(areas[0].centroid)
    return centroid.astype(np.int)


def visualize(data_dir):
    os.makedirs(data_dir + 'result/', exist_ok=True)

    paths = glob(data_dir + '*.jpg')
    names = [name.split('.')[1].split('/')[-1] for name in paths]


    for name in names:
        img = imread(data_dir + name + '.jpg', as_grey=True)
        gt = imread(data_dir + name + '_1.bmp', as_grey=True)

        coordinate = get_coordinate(gt)

        img_cropped = img[coordinate[0]-75:coordinate[0]+75, coordinate[1]-75:coordinate[1]+75]
        imsave(data_dir + 'result/' + name + '_cropped.eps', img_cropped)

        prediction = imread(data_dir + name + '_2.png', as_grey=True)

        vis_img = visulize_gt(img, gt)
        vis_img = visulize_gt(vis_img, prediction, mode='b')
        vis_img = vis_img[coordinate[0] - 75:coordinate[0] + 75, coordinate[1] - 75:coordinate[1] + 75]
        imsave(data_dir + 'result/' + name + '.eps', vis_img)


def visualize_feat():
    x_bottom = np.load('./feat_vis/x_bottom.npy')
    x_middle = np.load('./feat_vis/x_middle.npy')
    x_top = np.load('./feat_vis/x_top.npy')

    print(x_bottom.shape); print(x_middle.shape); print(x_top.shape)

    x_bottom = np.sum(x_bottom, axis=2)[0]
    x_middle = np.sum(x_middle, axis=2)[0]
    x_top = np.sum(x_top, axis=2)[0]

    x_bottom = x_bottom / np.sum(x_bottom) * 255
    x_middle = x_middle / np.sum(x_middle) * 255
    x_top = x_top / np.sum(x_top)* 255
    print(x_bottom.shape)
    for i in range(15):
        print(x_bottom[i])
        tmp = np.resize(x_bottom[i], (32, 32))
        print(tmp)
        imsave('./feat_vis/result/bottom_' + str(i) + '.bmp', (tmp*255).astype(np.uint8))

        tmp = np.resize(x_middle[i], (32, 32))
        imsave('./feat_vis/result/middle_' + str(i) + '.bmp', (tmp*255).astype(np.uint8))

        imsave('./feat_vis/result/top_' + str(i) + '.bmp', (x_top[i]*255).astype(np.uint8))




if __name__== '__main__':
    # visualize('./result_vis/')
    visualize_feat()