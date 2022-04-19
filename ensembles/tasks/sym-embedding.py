import os
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from . pytypes import *
from . model import BaseVAE, Decoder

class SymEmbedding(pl.LightningModule):
    """Task of embedding symbolic object into z-space"""

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(OGVAE, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img = batch[0]
        results = self.forward(real_img)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img = batch[0]
        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        recons = results[0]
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir ,
                                       "reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.sample_images(real_img.device)


    def sample_images(self, device):
        samples = self.model.sample(25,
                                    device)

        vutils.save_image(samples.cpu().data,
                        os.path.join(self.logger.log_dir ,
                                        "samples",
                                        f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                        normalize=True,
                        nrow=12)

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        if self.params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                         gamma = self.params['scheduler_gamma'])
            scheds.append(scheduler)

        return optims, scheds
