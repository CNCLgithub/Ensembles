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

        hidden_dims = [16, 32, 64, 64, 64, 64]
        in_channels = 1

        # Build Encoder
        for h_dim in hidden_dims:
            ogs_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 4, stride= 2, padding  = 1),
                    PrintLayer(),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.ogs_encoder = nn.Sequential(*ogs_modules, nn.Flatten())

        self.dk_encoder = nn.Sequential(
                nn.Linear(6,latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim ,latent_dim),
                nn.LeakyReLU()
        )

        self.encoder = nn.Sequential(
                nn.Linear(latent_dim,latent_dim),
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
                nn.Linear(latent_dim,latent_dim),
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
                    PrintLayer(),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.ogs_decoder = nn.Sequential(*ogs_modules)

        self.dk_decoder = nn.Sequential(
                nn.Linear(6,latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim,latent_dim),
                nn.LeakyReLU()
        )


    def encode(self, dk: Tensor, ogs: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        ogs=ogs.unsqueeze(1)
        dke = self.dk_encoder(dk)
        ogse = self.ogs_encoder(ogs)
        print(ogse.shape)
        result = torch.cat((dke, ogse), 1)
        result = self.encoder(result)
        #result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # print(result.shape)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        dkd, ogd = torch.split(result, self.latent_dim, 1)
        dkd = self.dk_decoder(dkd)
        ogd = self.ogs_decoder(ogd)
        #result = self.final_layer(result)
        return (dkd, ogd)

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

    def forward(self, dk: Tensor, ogs: Tensor) -> Tensor:
        mu, log_var = self.encode(dk, ogs)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

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
