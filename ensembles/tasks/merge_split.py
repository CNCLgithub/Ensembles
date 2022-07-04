import os
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils

from ensembles.pytypes import *
from ensembles.archs.vae import BaseVAE


# based off
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/09-normalizing-flows.html

class MergeSplit(pl.LightningModule):
    """Task of Merge or Split over z-space"""

    def __init__(self, flows, import_samples=8):
        """
        Args:
            flows: A list of flows (each a nn.Module) that should be applied on the images.
            import_samples: Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.import_samples = import_samples
        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        # Example input for visualizing the graph
        self.example_input_array = train_set[0][0].unsqueeze(dim=0)

    def forward(self, xs: Tensor) -> Tensor:
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(xs)

    def merge(self, xs: Tensor):
        # Given a batch of xs, return the merged representation z and ldj of the transformations
        z, ldj = xs, torch.zeros(xs.shape[0], device=self.device)
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_likelihood(self, xs: Tensor, return_ll = False):
        """Given a batch of object pairs, return the likelihood of those.
        """
        z, ldj = self.encode(xs)
        log_pz = self.prior.log_prob(z).sum(dim=[1, 2, 3])
        log_px = ldj + log_pz
        # Calculating bits per dimension

    def logpdf(self, xs: Tensor):
        """Given a batch of object pairs, return the likelihood of those.
        """
        z, ldj = self.encode(xs)
        log_pz = self.prior.log_prob(z).sum(dim=[1, 2, 3])
        log_px = ldj + log_pz
        return log_px

    def bpd(self, xs: Tensor):
        """ Bits per dimension
        """
        nll = -1 * self.logpdf(xs)
        bpd = nll * np.log2(np.exp(1)) / np.prod(xs.shape[1:])
        return bpd.mean()


    @torch.no_grad()
    def sample(self, batch_size = 1, constraint=None):
        """Sample a batch of images from the flow."""
        device = self.device
        # Sample latent representation from prior
        z_shape = [batch_size, *self.z_shape]
        if z_init is None:
            z = self.prior.sample(sample_shape=z_shape).to(device)
        # or condition on constraint (for full prior)
        else:
            z = z_init.to(device)

        # Transform z to x by inverting the flows
        ldj = torch.zeros(batch_size, device=device)
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        return z

    def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-3)
        optimizer = optim.Adam(self.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = self.params['scheduler_gamma'])
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self.bpd(batch[0])
        self.log("train_bpd", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.bpd(batch[0])
        self.log("val_bpd", loss)

    def test_step(self, batch, batch_idx):
        # Perform importance sampling during testing => estimate likelihood M times for each image
        samples = []
        for _ in range(self.import_samples):
            img_ll = self.logpdf(batch[0])
            samples.append(img_ll)
        img_ll = torch.stack(samples, dim=-1)

        # To average the probabilities, we need to go from log-space to exp, and back to log.
        # Logsumexp provides us a stable implementation for this
        img_ll = torch.logsumexp(img_ll, dim=-1) - np.log(self.import_samples)

        # Calculate final bpd
        bpd = -img_ll * np.log2(np.exp(1)) / np.prod(batch[0].shape[1:])
        bpd = bpd.mean()

        self.log("test_bpd", bpd)
    def forward(self, dk: Tensor) -> Tensor:
        return self.model(dk)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        dk= batch[0]
        results = self.forward(dk)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        #dk, ogs = batch
        dk = batch[0]
        print("validation step")
        print(type(self.forward(dk)))
        results = self.forward(dk)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)


        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)



       	results = results[1].unsqueeze(1)
        vutils.save_image(results.data,
                          os.path.join(self.logger.log_dir ,
                                       "reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=False,
                          nrow=6)
        vutils.save_image(ogs.unsqueeze(1).data,
                          os.path.join(self.logger.log_dir ,
                                       "reconstructions",
                                       f"gt_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=False,
                          nrow=6)
        #self.sample_ogs(real_img.device)

