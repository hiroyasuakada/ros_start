import os
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time
import cv2

from dataset import LSTMDataset



class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3),
                       nn.InstanceNorm2d(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),

            nn.Conv2d(3, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

        # initialize weights
        self.model.apply(self._init_weights)

    def forward(self, input_img):
        out = self.model(input_img)
        return out

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

        # initialize weights
        self.model.apply(self._init_weights)

    def forward(self, input_img):
        out = self.model(input_img)
        return out

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)

# =====================================================================================================================

# class ConvLSTMCell(nn.Module):
#     def __init__(self, input_channels, hidden_channels, kernel_size):
#         super(ConvLSTMCell, self).__init__()
#
#         assert hidden_channels % 2 == 0
#
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.kernel_size = kernel_size
#         self.num_features = 4
#
#         self.padding = int((kernel_size - 1) / 2)
#
#         self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
#         self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
#         self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
#         self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
#         self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
#         self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
#         self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
#         self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
#
#         self.Wci = None
#         self.Wcf = None
#         self.Wco = None
#
#     def forward(self, x, h, c):
#         ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
#         cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
#         cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
#         co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
#         ch = co * torch.tanh(cc)
#         return ch, cc
#
#     def init_hidden(self, batch_size, hidden, shape):
#         if self.Wci is None:
#             self.Wci = torch.tensor(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
#             self.Wcf = torch.tensor(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
#             self.Wco = torch.tensor(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
#         else:
#             assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
#             assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
#         return (torch.tensor(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(device),
#                 torch.tensor(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(device))
#
#
# class ConvLSTM(nn.Module):
#     # input_channels corresponds to the first input feature map
#     # hidden state is a list of succeeding lstm layers.
#     def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
#         super(ConvLSTM, self).__init__()
#         self.input_channels = [input_channels] + hidden_channels
#         self.hidden_channels = hidden_channels
#         self.kernel_size = kernel_size
#         self.num_layers = len(hidden_channels)
#         self.step = step
#         self.effective_step = effective_step
#         self._all_layers = []
#         for i in range(self.num_layers):
#             name = 'cell{}'.format(i)
#             cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
#             setattr(self, name, cell)
#             self._all_layers.append(cell)
#
#     def forward(self, _input):
#         internal_state = []
#         outputs = []
#         for step in range(self.step):
#             x = _input
#             for i in range(self.num_layers):
#                 # all cells are initialized in the first step
#                 name = 'cell{}'.format(i)
#                 if step == 0:
#                     bsize, _, height, width = x.size()
#                     (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
#                                                              shape=(height, width))
#                     internal_state.append((h, c))
#
#                 # do forward
#                 (h, c) = internal_state[i]
#                 x, new_c = getattr(self, name)(x, h, c)
#                 internal_state[i] = (x, new_c)
#             # only record effective steps
#             if step in self.effective_step:
#                 outputs.append(x)
#
#         return outputs, (x, new_c)
# =====================================================================================================================



class CNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(CNNEncoder, self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)




        )

        # initialize weights
        self.model.apply(self._init_weights)

    def forward(self, input_img):
        out = self.model(input_img)
        return out

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim,
                 hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size (int, int):
            height x width of input tensor
        input_dim (int):
            number of channels of input tensor
        hidden_dim (int):
            number of channels of hidden state
        kernel_size (int):
            size of filter kernel
        bias (bool):
            add bias or not
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        """
        注意输入输出的维度，是为了混合卷积，加快运算速度
        LSTM: gate_size = 4 * hidden_size
        GRU: gate_size = 3 * hidden_size
        """
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        input_tensor (Tensor):
            input x
        cur_state (Tensor, Tensor):
            h_out, c_out: the output of the previous state
        return：h_next, c_next
        """
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # ConvLSTM: concatenate to speed up
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, cuda_use):
        if cuda_use:
            return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device),
                    torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))
        else:
            return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device),
                    torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))


class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size,
                 num_layers, batch_first=False, bias=True,
                 return_all_layers=False, cuda_use=False):
        """
        Initialize ConvLSTM network.
        Parameters
        ----------
        input_size (int, int):
            height and width of input tensor
        input_dim (int):
            number of channels of input tensor
        hidden_dim (int):
            number of channels of hidden state
        kernel_size (int):
            size of filter kernel
        num_layers (int):
            number of ConvLSTMCells
        batch_first (bool):
            batch first or not
        bias (bool):
            add bias or not
        return_all_layers (bool):
            return all the layers or not
        cuda_use (bool):
            use GPU(s) or not
        """

        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # make sure kernel_size and hidden_dim are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if len(kernel_size) != num_layers or len(hidden_dim) != num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.cuda_use = cuda_use

        cell_list = []

        # add LSTM Unit tu cell_list
        for i in range(0, self.num_layers):
            if i == 0:
                cur_input_dim = self.input_dim
            else:
                cur_input_dim = hidden_dim[i - 1]

            if self.cuda_use:
                cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                              input_dim=cur_input_dim,
                                              hidden_dim=self.hidden_dim[i],
                                              kernel_size=self.kernel_size[i],
                                              bias=self.bias).to(device))
            else:
                cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                              input_dim=cur_input_dim,
                                              hidden_dim=self.hidden_dim[i],
                                              kernel_size=self.kernel_size[i],
                                              bias=self.bias))
            self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape
                (t, b, c, h, w) or
                (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        return: last_state_list, layer_output
        """


        # (t, b, c, h, w) -> (b, t, c, h, w)
        if not self.batch_first:
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), cuda_use=self.cuda_use)

            layer_output_list = []
            last_state_list = []

            seq_len = input_tensor.size(1)
            cur_layer_input = input_tensor

            for layer_idx in range(self.num_layers):

                h, c = hidden_state[layer_idx]
                output_inner = []
                for t in range(seq_len):
                    h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                     cur_state=[h, c])
                    output_inner.append(h)

                layer_output = torch.stack(output_inner, dim=1)
                cur_layer_input = layer_output

                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

            if not self.return_all_layers:
                layer_output_list = layer_output_list[-1:]
                last_state_list = last_state_list[-1:]

            return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, cuda_use=False):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, cuda_use=cuda_use))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print('device {}'.format(device))

    batch_size = 2
    window_size = 48
    step_size = 12
    train_dataset = LSTMDataset(window_size, step_size, is_train=True, is_condition=False)
    # train_dataset = UnalignedDataset(is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    data = iter(train_loader).next()
    print('===========================================================')
    print(data['A'].shape)
    print(data['A'][0].shape)
    # print(data['B'])
    print(data['path_A'])  # torch.Size([4, 8, 3, 128, 256])
    print(data['path_A'][0])  # torch.Size([8, 3, 128, 256])
    # print(data['path_B'])
    print('=====================')

    convlstm = ConvLSTM(input_size=(128, 256), input_dim=3, hidden_dim=[128, 128], kernel_size=(3, 3), num_layers=2,
                        batch_first=True, return_all_layers=True, cuda_use=True).to(device)

    layer_output_list, last_state_list = convlstm(data['A'].to(device))

    print(layer_output_list[0].shape)
    print(layer_output_list[1].shape)

    # # gradient check
    # convlstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5,
    #                     effective_step=[4]).to(device)
    # loss_fn = torch.nn.MSELoss()
    #
    # input = torch.tensor(torch.randn(1, 512, 64, 32)).to(device)
    # target = torch.tensor(torch.randn(1, 32, 64, 32)).double().to(device)
    #
    # output = convlstm(input)
    # output = output[0][0].double()
    # res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    # print(res)