import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvRNNCell(nn.Module):
    def __init__(self, in_channels, shape, num_filter, kernel_size=3):
        super(ConvRNNCell, self).__init__()
        padding = (kernel_size - 1) // 2
        self._shape = shape
        self._num_filter = num_filter
        self.conv_1 = nn.Conv2d(in_channels + num_filter, num_filter, kernel_size=1, padding=0, bias=True)
        self.conv_2 = nn.Conv2d(in_channels + num_filter, num_filter, kernel_size=kernel_size, padding=padding,bias=True)
	
    def forward(self, x, hidden):
        """forward process of ConvRNNCell"""
        combind = torch.cat((x, hidden), 1)
        reset_gate = F.sigmoid(self.conv_1(combind))
        
        reset_hidden = reset_gate * hidden
        
        combin_reset = torch.cat((x, reset_hidden), dim=1)
        new_hidden = F.relu(self.conv_2(combin_reset))  # TODO: applying tanh maybe a better choice
        return new_hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self._num_filter, self._shape[0], self._shape[1])).cuda(0)



class ConvRNN(nn.Module):
    def __init__(self, in_channels, shape, num_filter, kernel_size=3, num_layer=1, bidirectional=True):
        super(ConvRNN, self).__init__()
        self._in_channels = in_channels
        self._shape = shape
        self._num_filter = num_filter
        self._kernel_size = kernel_size
        self._num_layer = num_layer
        self._bidirectional = bidirectional
        self._padding = (self._kernel_size - 1) // 2
        self._cell_list = None
        self._forward_cell_list = None
        self._backward_cell_list = None

        if self._bidirectional:
            forward_cell_list = []
            backward_cell_list = []

            for idx in range(self._num_layer):
                forward_cell_list.append(
                    ConvRNNCell(self._in_channels, self._shape, self._num_filter, self._kernel_size))
                backward_cell_list.append(
                    ConvRNNCell(self._in_channels, self._shape, self._num_filter, self._kernel_size))
            self._forward_cell_list = nn.ModuleList(forward_cell_list)
            self._backward_cell_list = nn.ModuleList(backward_cell_list)
        else:
            cell_list = []
            for idx in range(self._num_layer):
                cell_list.append(ConvRNNCell(self._in_channels, self._shape, self._num_filter, self._kernel_size))
            self._cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state):
        curr_forward_x = x.transpose(0, 1)
        curr_backward_x = x.transpose(0, 1)

        forward_hidden_state, backward_hidden_state = hidden_state

        seq_len = curr_forward_x.size(0)

        if self._bidirectional:
            for idx_layer in range(self._num_layer):
                layer_forward_hidden = forward_hidden_state[idx_layer]
                layer_backward_hidden = backward_hidden_state[idx_layer]
                inner_forward_output = []
                inner_backward_output = []
                for t in range(seq_len):
                    # forward
                    layer_forward_hidden = self._forward_cell_list[idx_layer](curr_forward_x[t, ...],
                                                                              layer_forward_hidden)
                    inner_forward_output.append(layer_forward_hidden)
                    # backward
                    layer_backward_hidden = self._backward_cell_list[idx_layer](curr_backward_x[seq_len - t - 1, ...],
                                                                                layer_backward_hidden)
                    inner_backward_output.append(layer_backward_hidden)

                curr_forward_x = torch.cat(inner_forward_output, 0).view(curr_forward_x.size(0),
                                                                         *inner_forward_output[0].size())
                curr_backward_x = torch.cat(inner_backward_output, 0).view(curr_backward_x.size(0),
                                                                           *inner_backward_output[0].size())

            curr_forward_x = curr_forward_x.transpose(0, 1)
            curr_backward_x = curr_backward_x.transpose(0, 1)
            return curr_forward_x, curr_backward_x
        else:
            raise Exception('No single direction implemnted!')

    def init_hidden(self, batch_size):
        """
        init hidden state and cell state
        :param batch_size: 
        :return: 
        """
        if self._bidirectional:
            forward_hidden_state = []
            backward_hidden_state = []
            for i in range(self._num_layer):
                forward_hidden_state.append(self._forward_cell_list[i].init_hidden(batch_size))
                backward_hidden_state.append(self._backward_cell_list[i].init_hidden(batch_size))
            return tuple(forward_hidden_state), tuple(backward_hidden_state)
        else:
            hidden_state = []
            for i in range(self._num_layer):
                hidden_state.append(self._cell_list[i].init_hidden(batch_size))
            return tuple(hidden_state)
