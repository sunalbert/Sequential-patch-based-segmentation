import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from modules.TimeDistributed import TimeDistributed
from modules.ConvLSTM import ConvLSTM
from modules.ConvRNN import ConvRNN


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data.zero_())
    else:
        return tuple(repackage_hidden(v) for v in h)


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        """
        DownBlock
        :param in_planes: 
        :param out_planes: 
        """
        super(DownBlock, self).__init__()

        padding = (kernel_size-1) // 2

        self.t_conv_1 = TimeDistributed(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding,
                                                  stride=1, bias=True))
        self.t_conv_2 = TimeDistributed(nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=padding,
                                                  stride=1, bias=True))

    def forward(self, x):
        x = F.relu(self.t_conv_1(x))
        x = F.relu(self.t_conv_2(x))
        return x

    def init_weight(self):
        # TODO: weight init
        pass


class UpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, up_scale=2):
        """
        upsampling with deconvolution
        :param in_planes: 
        :param out_planes: 
        """
        super(UpBlock, self).__init__()
        self._in_planes = in_planes
        self._out_planes = out_planes
        self.t_deconv_1 = TimeDistributed(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=up_scale,
                                                             stride=up_scale, bias=True))
        self.conv_1 = TimeDistributed(nn.Conv2d(out_planes*2, out_planes, kernel_size=3, padding=1, stride=1, bias=True))
        self.conv_2 = TimeDistributed(nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=True))

    def forward(self, x, pre):
        x = self.t_deconv_1(x)
        x = torch.cat((pre, x), dim=2)
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        return x


class ConvRNNBlock(nn.Module):
    def __init__(self, batch_size, in_channels, shape, num_filter, kernel_size):
        super(ConvRNNBlock, self).__init__()
        self.conv_rnn = ConvRNN(in_channels, shape, num_filter, kernel_size)
        self.conv = TimeDistributed(nn.Conv2d(2 * num_filter, num_filter, kernel_size=1, bias=True))
        self._hidden_state = self.conv_rnn.init_hidden(batch_size)

    def forward(self, x):
        forward_out, backward_out = self.conv_rnn(x, self._hidden_state)
        out = torch.cat((forward_out, backward_out), dim=2)
        out = F.relu(self.conv(out))
        return out

    def reinit_hidden(self):
        self._hidden_state = repackage_hidden(self._hidden_state)