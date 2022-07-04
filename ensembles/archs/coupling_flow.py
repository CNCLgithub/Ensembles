import torch
from torch import nn
import torch.nn.functional as F

from ensembles.pytypes import *
# based on
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/09-normalizing-flows.html#References

def create_checkerboard_mask(n: Int, invert=False):
    x = torch.arange(n, dtype=torch.int32)
    mask = torch.fmod(x, 2)
    mask = mask.to(torch.float32).view(1, n)
    if invert:
        mask = 1 - mask
    return mask

def create_ABAC_mask(n: Int, invert = False):
    a_mask = torch.ones(1, n)
    b_mask = create_checkerboard_mask(n, invert = invert)
    return torch.cat((a_mask, b_mask), dims = 1)

class CouplingLayer(nn.Module):
    def __init__(self, network, mask):
        """Coupling layer inside a normalizing flow.

        Args:
            network: A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask: Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
        """
        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.zeros(1))
        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        self.register_buffer("mask", mask)

    def forward(self, z, ldj, reverse=False):
        """
        Args:
            z: Latent input to the flow
            ldj: The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse: If True, we apply the inverse of the layer.
        """
        # Apply network to masked input
        z_in = z * self.mask
        if orig_img is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(torch.cat([z_in, orig_img], dim=1))
        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1, 2, 3])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1, 2, 3])

        return z, ldj
