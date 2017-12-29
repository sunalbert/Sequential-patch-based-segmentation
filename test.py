"""
This file is just an example to show
how our method works. For convenience,
we directly calculate the mass center 
of the object according to the ground 
truth(in function: get_coor(mask, is_gt)) as the 
initialize point. 
"""
import torch
import os
import test_params
import numpy as np
from glob import glob
from math import floor
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.io import imsave, imread
from pre_process import direction_extract
from modules.Model import SkipConnecRNNModel
from skimage.measure import regionprops, label
from pre_process import set_patch
from  utils.post_process import post_process
from utils.statistic import dice_ratio, cd_xy
from utils.visualize import visulize_gt

HEIGHT = 296
WIDTH = 296
PATCH_HEIGHT = 32
PATCH_WIDTH = 32
PADDING_SHAPE = (436, 436)
directions = [(0, 1), (-1, 2), (-1, 1), (-2, 1), (-1, 0), (-2, -1), (-1, -1), (-1, -2)]

model_file = 'weights.pth'


def integrate(result, counter_map, patches, coordinate, direction):
    """
    integrate patch result into whole image
    :param result:
    :param patches: segmentation result np.array
    :param direction: 
    :param short_name: 
    :return: whole image segmentation result
    """
    seq_length = len(patches[0])
    coord = [coordinate[0], coordinate[1]]

    # positve direction
    for i in range(seq_length):
        coord[0] += direction[0] * 4
        coord[1] += direction[1] * 4
        set_patch(result, counter_map, patches[0][i], coord)

    coord = [coordinate[0], coordinate[1]]

    # negative direction
    for i in range(seq_length):
        coord[0] -= direction[0] * 4
        coord[1] -= direction[1] * 4
        set_patch(result, counter_map, patches[1][i], coord)


def find_red(img):
    coords = []
    H, W, C = img.shape
    for i in range(H):
        for j in range(W):
            if img[i][j][0] == 255 and img[i][j][1] == 0 and img[i][j][2] == 0:
                coords.append([i, j])
    return np.asarray(coords)


def get_coor(mask, is_gt = False):
    """
    get the coordinate of the center point_move
    """
    if is_gt:
        areas = regionprops(label(mask // 255))
        assert len(areas) == 1
        centroid = np.rint(areas[0].centroid)
        return centroid.astype(np.int)
    else:
        coords = find_red(mask)
        centroid = np.mean(coords, axis=0)
        centroid = np.floor(centroid).astype(np.int)
        return centroid


def inference():
    """
    Test the model on the testing data
    :return: 
    """
    os.makedirs(test_params.output_dir + '/whole_seg', exist_ok=True)
    net = SkipConnecRNNModel(img_shape=(32, 32), num_class=2, batch_size=2)
    net.load_state_dict(torch.load(model_file))
    net.cuda()
    net.eval()

    names = glob(test_params.test_data_dir + '*.jpg')
    names = [name.split('.')[0].split('/')[-1] for name in names]
    print(len(names))

    dices = {}
    cds = []

    # start predicting
    for name in names:
        print(name)
        img = imread(test_params.test_data_dir + name + '.jpg')
        gt_img = imread(test_params.test_data_dir + name + '_1.bmp')

        result = np.zeros(PADDING_SHAPE, dtype=np.float)
        counter_map = np.zeros(PADDING_SHAPE, dtype=np.float)

        new_img = np.zeros(PADDING_SHAPE, dtype=np.uint8)
        new_img[(PADDING_SHAPE[0] - HEIGHT) // 2:(PADDING_SHAPE[0] - HEIGHT) // 2 + 296,
        (PADDING_SHAPE[0] - WIDTH) // 2:(PADDING_SHAPE[0] - WIDTH) // 2 + 296] = img
        new_gt = np.zeros(PADDING_SHAPE, dtype=np.uint8)
        new_gt[(PADDING_SHAPE[0] - HEIGHT) // 2:(PADDING_SHAPE[0] - HEIGHT) // 2 + 296,
        (PADDING_SHAPE[0] - WIDTH) // 2:(PADDING_SHAPE[0] - WIDTH) // 2 + 296] = gt_img

        coordinate = get_coor(new_gt, is_gt=True)

        for direction in directions:
            img_seqs, label_seqs = direction_extract(new_img, new_gt, coordinate, direction)
            for i in range(15):
                imsave('label_' + str(i) + '.bmp', label_seqs[1][i])
            img_seqs = np.asarray(img_seqs).astype(np.float32)
            img_seqs = np.expand_dims(img_seqs,axis=2)
            img_seqs /= 255
            assert img_seqs.shape == (2, 15, 1, 32, 32)
            img_seqs = torch.from_numpy(img_seqs)
            img_seqs = Variable(img_seqs.cuda(), volatile=True)
            net.reinit_hidden()

            logits, x_bottom, x_middle, x_top = net(img_seqs)
            x_bottom, x_middle, x_top = x_bottom.data.cpu().numpy(), x_middle.data.cpu().numpy(), x_top.data.cpu().numpy()
            np.save('x_bottom.npy', x_bottom); np.save('x_middle.npy', x_middle); np.save('x_top.npy', x_top)

            probs = F.sigmoid(logits)
            masks = (probs > 0.5).float()
            masks = masks.data.cpu().numpy()
            masks = np.squeeze(masks, axis=2)

            integrate(result, counter_map, masks, coordinate, direction)

        result /= counter_map
        result = result * 255

        result_img = result[(PADDING_SHAPE[0] - HEIGHT) // 2:(PADDING_SHAPE[0] - HEIGHT) // 2 + 296,
                     (PADDING_SHAPE[0] - WIDTH) // 2:(PADDING_SHAPE[0] - WIDTH) // 2 + 296]
        imsave(test_params.output_dir + 'whole_seg/' + name + '_1.png', result_img.astype(np.uint8))

        result_img = post_process(result_img.astype(np.uint8))
        imsave(test_params.output_dir + 'whole_seg/' + name + '_2.png', result_img.astype(np.uint8))
        cd = cd_xy(result_img.astype(np.uint8), gt_img)
        cds.append(cd)

        dices[name] = dice_ratio(result_img // 255, gt_img // 255)

        result_img = visulize_gt(result_img.astype(np.uint8), gt_img)
        imsave(test_params.output_dir + 'whole_seg/' + name + '_3.png', result_img.astype(np.uint8))

    print(dices)
    total = 0
    tmp_dices = []
    for key, value in dices.items():
        print(key + ' ' + str(value))
        total += value
        tmp_dices.append(value)
    print(total / len(dices))

    cds = np.asarray(cds)
    print(cds)
    print(np.mean(cds, axis=0))
    print(np.var(cds, axis=0))
    tmp_dices = np.asarray(tmp_dices)
    print(np.mean(tmp_dices))
    print(np.var(tmp_dices))


if __name__ == '__main__':
    inference()




