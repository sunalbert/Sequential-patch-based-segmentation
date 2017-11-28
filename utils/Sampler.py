import random
from torch.utils.data.sampler import Sampler


class RandomSamplerWithLength(Sampler):
    def __init__(self, data, length):
        """
        Random Sampler with fixed length
        :param data: dataset
        :param length: num of data
        """
        self.num_samples = length
        self.len_data = len(data)

    def __iter__(self):
        l = list(range(self.len_data))
        random.shuffle(l)
        l = l[0:self.num_samples]
        return iter(l)

    def __len__(self):
        return self.num_samples