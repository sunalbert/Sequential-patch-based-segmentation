import torch
import os
import test_params
import numpy as np
from glob import glob
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.io import imsave, imread
from pre_process import direction_extract
from modules.Model import SkipConnecRNNModel
from skimage.measure import regionprops, label
from pre_process import set_patch
from  utils.post_process import post_process
from utils.statistic import dice_ratio
from utils.visualize import visulize_gt

HEIGHT = 296
WIDTH = 296
PATCH_HEIGHT = 32
PATCH_WIDTH = 32
PADDING_SHAPE = (436, 436)
directions = [(0, 1), (-1, 2), (-1, 1), (-2, 1), (-1, 0), (-2, -1), (-1, -1), (-1, -2)]

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

test_data_dir = './point_move/'
output_dir = './point_move/'
model_dir = test_params.output_dir

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


def inference(coordinate, total_dice):
    """
    Test the model on the testing data
    :return:
    """
    os.makedirs(test_data_dir + '/whole_seg', exist_ok=True)
    net = SkipConnecRNNModel(img_shape=(32, 32), num_class=2, batch_size=2)

    model_file = model_dir + 'snap/final.pth'
    print(model_file)
    net.load_state_dict(torch.load(model_file))
    net.cuda()
    net.eval()

    names = glob(test_data_dir + '*.jpg')
    print(names)
    names = [name.split('.')[1].split('/')[-1] for name in names]
    print(len(names))
    print(names)

    dices = {}

    # start predicting
    for name in names:
        print(name)
        img = imread(test_data_dir + name + '.jpg')
        gt_img = imread(test_data_dir + name + '_1.bmp')

        result = np.zeros(PADDING_SHAPE, dtype=np.float)
        counter_map = np.zeros(PADDING_SHAPE, dtype=np.float)

        new_img = np.zeros(PADDING_SHAPE, dtype=np.uint8)
        new_img[(PADDING_SHAPE[0] - HEIGHT) // 2:(PADDING_SHAPE[0] - HEIGHT) // 2 + 296,
        (PADDING_SHAPE[0] - WIDTH) // 2:(PADDING_SHAPE[0] - WIDTH) // 2 + 296] = img
        new_gt = np.zeros(PADDING_SHAPE, dtype=np.uint8)
        new_gt[(PADDING_SHAPE[0] - HEIGHT) // 2:(PADDING_SHAPE[0] - HEIGHT) // 2 + 296,
        (PADDING_SHAPE[0] - WIDTH) // 2:(PADDING_SHAPE[0] - WIDTH) // 2 + 296] = gt_img

        # coordinate = get_coor(new_gt, is_gt=True)
        print(coordinate)

        for direction in directions:
            img_seqs, _ = direction_extract(new_img, new_gt, coordinate, direction)
            img_seqs = np.asarray(img_seqs).astype(np.float32)
            img_seqs = np.expand_dims(img_seqs,axis=2)
            img_seqs /= 255
            assert img_seqs.shape == (2, 15, 1, 32, 32)
            img_seqs = torch.from_numpy(img_seqs)
            img_seqs = Variable(img_seqs.cuda(), volatile=True)
            net.reinit_hidden()

            logits = net(img_seqs)
            probs = F.sigmoid(logits)
            masks = (probs > 0.5).float()
            masks = masks.data.cpu().numpy()
            masks = np.squeeze(masks, axis=2)

            integrate(result, counter_map, masks, coordinate, direction)

        result /= counter_map
        result = result * 255

        result_img = result[(PADDING_SHAPE[0] - HEIGHT) // 2:(PADDING_SHAPE[0] - HEIGHT) // 2 + 296,
                     (PADDING_SHAPE[0] - WIDTH) // 2:(PADDING_SHAPE[0] - WIDTH) // 2 + 296]
        imsave(output_dir + 'whole_seg/' + name + '_1.png', result_img.astype(np.uint8))

        result_img = post_process(result_img.astype(np.uint8))
        imsave(output_dir + 'whole_seg/' + name + '_2.png', result_img.astype(np.uint8))

        tmp_dice = dice_ratio(result_img // 255, gt_img // 255)
        dices[name] = tmp_dice
        total_dice.append(tmp_dice)

        result_img = visulize_gt(result_img.astype(np.uint8), gt_img)
        imsave(output_dir + 'whole_seg/' + name + '_3.png', result_img.astype(np.uint8))

    print(dices)
    total = 0
    for key, value in dices.items():
        print(key + ' ' + str(value))
        total += value


if __name__ == '__main__':
    total_dice = []
    coordinates = [(224, 227), (224, 232), (224, 237), (224, 242), (224, 247), (229, 227), (229, 232), (229, 237), (229, 242), (229, 247), (234, 227), (234, 232), (234, 237), (234, 242), (234, 247), (239, 227), (239, 232), (239, 237), (239, 242), (239, 247), (244, 227), (244, 232), (244, 237), (244, 242), (244, 247)]
    for coordinate in coordinates:
        inference(coordinate, total_dice=total_dice)
    print(total_dice)




