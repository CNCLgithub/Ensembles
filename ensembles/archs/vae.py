import torch
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod

from ensembles.pytypes import *

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class BetaVAE(BaseVAE):
    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 latent_dim: int,
                 beta: int = 4,
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        ogs_modules = []

        hidden_dims = [16, 16, 32, 64, 64, 64, 64]
        in_channels = 1

        # Build Encoder
        for h_dim in hidden_dims:
            ogs_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 4, stride= 2, padding  = 1),
                    #PrintLayer(),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.ogs_encoder = nn.Sequential(*ogs_modules, nn.Flatten())

        self.dk_encoder = nn.Sequential(
                nn.Linear(20,latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim ,latent_dim),
                nn.LeakyReLU()
        )
        self.encoder = nn.Sequential(
                nn.Linear(2*latent_dim,latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim,latent_dim),
                nn.LeakyReLU()
        )



        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)


        # Build Decoder
        ogs_modules = []

        self.decoder = nn.Sequential(
                nn.Linear(latent_dim,latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim, 2*latent_dim),
                nn.LeakyReLU()
        )


        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            ogs_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride = 2,
                                       padding=1),
                    #PrintLayer(),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.ogs_decoder = nn.Sequential(*ogs_modules,
                           nn.Conv2d(hidden_dims[-1], out_channels= 4,
                                     kernel_size= 3, padding= 1),
                           #nn.LeakyReLU()
                           nn.Sigmoid()
                                     #PrintLayer()
                                     )


        self.dk_decoder = nn.Sequential(
                nn.Linear(latent_dim,latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim, 20),
                nn.Tanh()
        )


    def encode(self, dk: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        #ogs=ogs.unsqueeze(1)
        #print(type(dk))
        dke = self.dk_encoder(dk)
        #ogse = self.ogs_encoder(ogs)
        #result = torch.cat((dke, ogse), 1)
        #print(result.shape)
        # print(result.shape)
        mu = self.fc_mu(dke)
        log_var = self.fc_var(dke)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        #print(result.shape)
        #print(torch.split(result, self.latent_dim, 1))
        #dkd, ogd = torch.split(result, self.latent_dim, 1)
        #ogd = ogd.view((z.shape[0], -1, 2, 2))
        dkd = self.dk_decoder(z)
        #ogd = self.ogs_decoder(ogd)
        #ogd=ogd.view(z.shape[0],128,128)
        #result = self.final_layer(result)
        return dkd

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, dk: Tensor) -> Tensor:
        mu, log_var = self.encode(dk)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), dk, mu, log_var]

    def loss_function(self,
                      recons_dk, dk, mu, log_var,
                      **kwargs) -> dict:
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        #recons_loss =F.mse_loss(recons_dk, dk)/(dk.shape[-1]) + F.mse_loss(recons_ogs, ogs)/(ogs.shape[-1] * ogs.shape[-2])
        recons_loss =F.mse_loss(recons_dk, dk)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.beta * kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
