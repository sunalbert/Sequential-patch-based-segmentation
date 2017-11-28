import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from modules.blocks import DownBlock, UpBlock, ConvLSTMBlock, ConvRNNBlock
from modules.TimeDistributed import TimeDistributed


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        num = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.norm_(0, math.sqrt(2. / num))
        if m.bias:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        pass


class SkipConnecModel(nn.Module):
    def __init__(self, img_shape, num_class, batch_size):
        """
        SkipConnection Model,which is a U-net like net structure
        :param num_class: 
        """
        super(SkipConnecModel, self).__init__()
        self._num_class = num_class

        self.downblock_1 = DownBlock(1, 32, 3)
        self.t_max_pool_1 = TimeDistributed(nn.MaxPool2d(2, 2))

        self.downblock_2 = DownBlock(32, 64, 3)
        self.t_max_pool_2 = TimeDistributed(nn.MaxPool2d(2, 2))

        self.downblock_3 = DownBlock(64, 128, 3)

        self.bottom_clstm = ConvLSTMBlock(batch_size, 128, (img_shape[0] // 4, img_shape[1] // 4), 128, 3)
        self.middle_clstm = ConvLSTMBlock(batch_size, 64, (img_shape[0] // 2, img_shape[1] // 2), 64, 3)
        self.top_clstm = ConvLSTMBlock(batch_size, 32, img_shape, 32, 3)

        self.upblock_1 = UpBlock(128, 64, up_scale=2)
        self.upblock_2 = UpBlock(64, 32, up_scale=2)

        self.final_conv = TimeDistributed(nn.Conv2d(32, num_class-1, kernel_size=1, stride=1))

    def forward(self, x):
        x_pool_1_pre = self.downblock_1(x)
        x = self.t_max_pool_1(x_pool_1_pre)  # (batch_size seq 32 H/2 W/2)

        x_pool_2_pre = self.downblock_2(x)
        x = self.t_max_pool_2(x_pool_2_pre) # (batch_size seq 64 H/4 W/4)

        x = self.downblock_3(x)  # (batch_size seq 128 H/4 W/4)

        x = self.bottom_clstm(x)

        x = self.upblock_1(x, x_pool_2_pre) # (batch_size seq 64 H/2 W/2)

        x = self.middle_clstm(x)

        x = self.upblock_2(x, x_pool_1_pre)

        x = self.top_clstm(x)

        x = self.final_conv(x)

        return x

    def reinit_hidden(self):
        """
        reinit the hidden state for all the lstm module
        :return: void
        """
        self.bottom_clstm.reinit_hidden()
        self.middle_clstm.reinit_hidden()
        self.top_clstm.reinit_hidden()


class SkipConnecRNNModel(nn.Module):
    def __init__(self, img_shape, num_class, batch_size):
        """
        SkipConnection Model,which is a U-net like net structure
        :param num_class: 
        """
        super(SkipConnecRNNModel, self).__init__()
        self._num_class = num_class

        self.downblock_1 = DownBlock(1, 32, 3)
        self.t_max_pool_1 = TimeDistributed(nn.MaxPool2d(2, 2))

        self.downblock_2 = DownBlock(32, 64, 3)
        self.t_max_pool_2 = TimeDistributed(nn.MaxPool2d(2, 2))

        self.downblock_3 = DownBlock(64, 128, 3)

        self.bottom_clstm = ConvRNNBlock(batch_size, 128, (img_shape[0] // 4, img_shape[1] // 4), 128, 3)
        self.middle_clstm = ConvRNNBlock(batch_size, 64, (img_shape[0] // 2, img_shape[1] // 2), 64, 3)
        self.top_clstm = ConvRNNBlock(batch_size, 32, img_shape, 32, 3)

        self.upblock_1 = UpBlock(128, 64, up_scale=2)
        self.upblock_2 = UpBlock(64, 32, up_scale=2)

        self.final_conv = TimeDistributed(nn.Conv2d(32, num_class-1, kernel_size=1, stride=1))

    def forward(self, x):
        x_pool_1_pre = self.downblock_1(x)
        x = self.t_max_pool_1(x_pool_1_pre)  # (batch_size seq 32 H/2 W/2)

        x_pool_2_pre = self.downblock_2(x)
        x = self.t_max_pool_2(x_pool_2_pre) # (batch_size seq 64 H/4 W/4)

        x = self.downblock_3(x)  # (batch_size seq 128 H/4 W/4)

        x_bottom = self.bottom_clstm(x)

        x = self.upblock_1(x_bottom, x_pool_2_pre) # (batch_size seq 64 H/2 W/2)

        x_middle = self.middle_clstm(x)

        x = self.upblock_2(x_middle, x_pool_1_pre)

        x_top = self.top_clstm(x)

        x = self.final_conv(x_top)

        return x, x_bottom, x_middle, x_top

    def reinit_hidden(self):
        """
        reinit the hidden state for all the lstm module
        :return: void
        """
        self.bottom_clstm.reinit_hidden()
        self.middle_clstm.reinit_hidden()
        self.top_clstm.reinit_hidden()


class MultiLevelModel(nn.Module):
    def __init__(self, bacth_size, img_shape, num_class):
        super(MultiLevelModel, self).__init__()
        pass

    def forward(self, x):
        pass

    def reinit_hidden(self):
        """
        reinit the hidden state for all the lstm module
        :return: void
        """
        self.bottom_clstm.reinit_hidden()
        self.middle_clstm.reinit_hidden()
        self.top_clstm.reinit_hidden()

