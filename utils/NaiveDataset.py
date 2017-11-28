import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from skimage.io import imsave, imshow


def image_to_tensor(img_seq, mean=0, std=1.):
    """
    convert the image sequence into torch tensor
    do normalization
    :param img_seq: (seq H W)
    :param mean: 
    :param std: 
    :return: 
    """
    img_seq = np.expand_dims(img_seq, axis=1)
    assert img_seq.shape == (15, 1, 32, 32)

    img_seq = img_seq.astype(np.float32)
    img_seq = (img_seq - mean) / std
    tensor = torch.from_numpy(img_seq)  ##.float()
    return tensor


def label_to_tensor(label, threshold=0.5):
    label = (label > threshold).astype(np.float32)
    tensor = torch.from_numpy(label).type(torch.FloatTensor)
    return tensor


def tensor_to_image(tensor, mean=0, std=1):
    image = tensor.numpy()
    image = image * std + mean
    image = image.astype(dtype=np.uint8)
    return image


def tensor_to_label(tensor):
    label = tensor.numpy() * 255
    label = label.astype(dtype=np.uint8)
    return label


class CustomizedDataset(Dataset):
    def __init__(self, path, transforms, mode):
        """
        Customized dataset wrapper for cvpr-18
        :param path: 
        :param transforms: 
        :param mode: 
        """
        super(CustomizedDataset, self).__init__()
        self._path = path
        self._transforms = transforms
        self._mode = mode

        self._load_data()

    def _load_data(self):
        if self._mode == 'train':
            self._train_data = np.load(self._path + 'train_img.npy')
            self._train_labels = np.load(self._path + 'train_label.npy')
            self._num_samples = self._train_data.shape[0]
        elif self._mode == 'val':
            self._val_data = np.load(self._path + 'val_img.npy')
            self._val_labels = np.load(self._path + 'val_label.npy')
            self._num_samples = self._val_data.shape[0]
        else:
            self._test_data = np.load(self._path + 'test_img.npy')
            self._num_samples = self._test_data.shape[0]

    def get_train_item(self, index):
        img_seq = self._train_data[index]
        label_seq = self._train_labels[index]

        img_seq = img_seq.astype(np.float32) / 255
        label_seq = label_seq.astype(np.float32) / 255

        # transform into torch tensor
        for t in self._transforms:
            img_seq, label_seq = t(img_seq, label_seq)
        img_seq = image_to_tensor(img_seq, mean=0, std=1)
        label_seq = label_to_tensor(label_seq)
        return img_seq, label_seq, index

    def get_val_item(self, index):
        img_seq = self._val_data[index]
        label_seq = self._val_labels[index]

        img_seq = img_seq.astype(np.float32) / 255
        label_seq = label_seq.astype(np.float32) / 255

        # transform into torch tensor
        for t in self._transforms:
            img_seq, label_seq = t(img_seq, label_seq)
        img_seq = image_to_tensor(img_seq)
        label_seq = label_to_tensor(label_seq)
        return img_seq, label_seq, index

    def get_test_item(self, index):
        pass

    def __getitem__(self, index):
        if self._mode == 'train':
            return self.get_train_item(index)
        elif self._mode == 'val':
            return self.get_val_item(index)
        else:
            return self.get_test_item(index)

    def __len__(self):
        return self._num_samples


def run_check_dataset():
    dataset = CustomizedDataset(path='/data/jinquan/data/cvpr_kidney/input/vanilla/', transforms=[], mode='train')
    print('load complete')
    for n in range(2):
        img_seq, label_seq, index = dataset[n]
        img_seq = tensor_to_image(img_seq, std=255)
        label_seq = tensor_to_label(label_seq)
        print(img_seq.shape)
        print(label_seq.shape)
        count = 0
        for img, label in zip(img_seq, label_seq):
            imsave('./test/img_' + str(n) + '_' + str(count) + '_1.png', img.squeeze())
            imsave('./test/img_' + str(n) + '_' + str(count) + '_2.png', label)
            count += 1

if __name__ == '__main__':
    run_check_dataset()
