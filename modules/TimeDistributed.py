import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module):
        """
        Given an input shaped like (batch, time_step, [rest]), and a Module
        TimeDistributed Module reshap the input to be (batch*time_step, [rest])
        :param module: 
        :return: 
        """
        super(TimeDistributed, self).__init__()
        self._module = module

    def forward(self, x):
        batch, time, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
        shape = x.size()
        x = x.view(batch*time, C, H, W)  #
        x = self._module(x)
        C, H, W = x.size(1), x.size(2), x.size(3)
        new_shape = (batch, time, C, H, W)
        x = x.view(new_shape)
        return x
