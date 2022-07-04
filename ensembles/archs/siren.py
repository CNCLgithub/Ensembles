# Several chunks cribbed from
# https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
import math
import torch
from torch import nn
import torch.nn.functional as F
# from einops import rearrange
# from siren_pytorch import SirenNet, SirenWrapper

from ensembles.pytypes import *

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(self, dim_in:int, dim_out:int,
                 w0:float = 1., w_std:float = 1.,
                 bias = True, activation = False):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias = bias)
        self.activation = Sine(w0) if activation else nn.Identity()

    def forward(self, x:Tensor):
        x =  self.linear(x)
        return self.activation(x)
    # def init_(self, weight, bias, c, w0):
    #     dim = self.dim_in

    #     w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
    #     weight.uniform_(-w_std, w_std)

    #     if exists(bias):
    #         bias.uniform_(-w_std, w_std)

class SirenNet(nn.Module):
    def __init__(self,
                 theta_in:int,
                 theta_hidden:int,
                 theta_out:int,
                 depth:int,
                 w0 = 1.,
                 w0_initial = 30.,
                 c = 6.0,
                 use_bias = True, final_activation = False):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.layers.append(Siren(
            dim_in = theta_in,
            dim_out = theta_hidden,
            w0 = w0_initial,
            w_std = 1.0 / theta_in,
            bias = use_bias,
        ))
        w_std = math.sqrt(c / theta_hidden) / w0
        for _ in range(depth - 1):
            self.layers.append(Siren(
                dim_in = theta_hidden,
                dim_out = theta_hidden,
                w0 = w0,
                w_std = w_std,
                bias = use_bias,
            ))
        self.last_layer = Siren(dim_in = theta_hidden, dim_out = theta_out, w0 = w0,
                                bias = use_bias, activation = final_activation)

    def forward(self, x:Tensor, phi:Tensor):
        for i in range(self.depth):
            x = self.layers[i](x)
            x *= phi[i]
        return self.last_layer(x)

class Modulator(nn.Module):
    def __init__(self, dim_in: int, dim_hidden:int , depth: int):
        super().__init__()
        self.layers = nn.ModuleList([])
        for ind in range(depth):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z:Tensor):
        x = z
        hiddens = []
        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            # TODO: remove extra cat for last step
            x = torch.cat((x, z), dim = -1)

        return hiddens

class ImplicitNeuralModule(nn.Module):

    def __init__(self,
                 theta_input: int,
                 theta_out: int,
                 theta_hidden: int = 256,
                 depth: int = 5):
        super().__init__()
        self.theta = SirenNet(theta_input, theta_hidden, theta_out, depth)
        self.psi = Modulator(theta_hidden, theta_hidden, depth)

    def forward(self, qs:Tensor, xs:Tensor):
        # TODO: generalize chunking `xs` to different sizes?
        es, ks = torch.chunk(xs, 2, dim = 1)
        # ks = rearrange([qs, ks], '2 b d -> b (2 d)')
        ks = torch.cat((qs, ks), dim = 1)
        phi = self.psi(es)
        return self.theta(ks, phi)
