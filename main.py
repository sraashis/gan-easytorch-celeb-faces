import argparse
import os

import torch
import torchvision.transforms as tmf
import torchvision.utils as vutils
from PIL import Image as IMG
from easytorch import EasyTorch
from easytorch.core.metrics import ETAverages
from easytorch.core.nn import ETTrainer, ETDataset
from easytorch.utils.defaultargs import ap

import dataspecs as dspec
from models import Generator, Discriminator

sep = os.sep


class CelebDataset(ETDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.image_size = 64

    def __getitem__(self, index):
        dset, file = self.indices[index]
        dt = self.dataspecs[dset]
        img = self.transforms(IMG.open(dt['data_dir'] + sep + file))
        return {'indices': self.indices[index], 'input': img}

    @property
    def transforms(self):
        return tmf.Compose(
            [tmf.Resize(self.image_size), tmf.CenterCrop(self.image_size),
             tmf.ToTensor(), tmf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class GANTrainer(ETTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.REAL_LABEL = 1
        self.FAKE_LBL = 0
        self.adversarial_loss = torch.nn.BCELoss()

    def _init_nn_model(self):
        self.nn['gen'] = Generator(self.args['num_channel'], self.args['latent_size'], self.args['map_gen_size'])
        self.nn['dis'] = Discriminator(self.args['num_channel'], self.args['map_dis_size'])

    def _init_optimizer(self):
        self.optimizer['gen'] = torch.optim.Adam(self.nn['gen'].parameters(), lr=self.args['learning_rate'],
                                                 betas=(0.5, 0.999))
        self.optimizer['dis'] = torch.optim.Adam(self.nn['gen'].parameters(), lr=self.args['learning_rate'],
                                                 betas=(0.5, 0.999))

    def training_iteration(self, batch):
        real_data = batch['input'].to(self.device['gpu'])
        N = real_data.size(0)

        noise = torch.randn(N, self.args['latent_size'], 1, 1, device=self.device['gpu'])
        fake_data = self.nn['gen'](noise).detach()

        real_label = torch.full((N, 1, 1, 1), self.REAL_LABEL, dtype=torch.float, device=self.device['gpu'])
        fake_label = torch.full((N, 1, 1, 1), self.FAKE_LBL, dtype=torch.float, device=self.device['gpu'])
        """
        Train Discriminator
        """
        self.optimizer['dis'].zero_grad()
        # Train on Real Data
        prediction_real = self.nn['dis'](real_data)
        # Calculate loss and back-propagate
        loss_real = self.adversarial_loss(prediction_real, real_label)
        loss_real.backward()

        # Train on Fake Data
        prediction_fake = self.nn['dis'](fake_data)
        # Calculate error and back-propagate
        loss_fake = self.adversarial_loss(prediction_fake, fake_label)
        loss_fake.backward()

        # Update weights with gradients
        self.optimizer['dis'].step()
        d_loss = loss_real + loss_fake

        """
        Train Generator
        """
        self.nn['gen'].zero_grad()
        # Sample noise and generate fake data
        prediction = self.nn['dis'](fake_data)
        # Calculate error and back-propagate
        g_loss = self.adversarial_loss(prediction, real_label)
        g_loss.backward()
        # Update weights with gradients
        self.optimizer['gen'].step()

        losses = self.new_averages()
        losses.add(d_loss.item(), len(batch['input']), index=0)
        losses.add(g_loss.item(), len(batch['input']), index=1)
        return {'averages': losses, 'fake_images': fake_data}

    def _on_iteration_end(self, i, ep, it):
        if ep > 5 and i % 100 == 0:  # Save every 20th batch from each epoch
            grid = vutils.make_grid(it['fake_images'], padding=2, normalize=True)
            vutils.save_image(grid, f"{self.cache['log_dir']}{sep}{i}.png")

    def new_averages(self):
        return ETAverages(num_averages=2)

    def reset_fold_cache(self):
        self.cache['training_log'] = ['D_LOSS,G_LOSS']


ap = argparse.ArgumentParser(parents=[ap], add_help=False)
ap.add_argument('-nz', '--latent_size', default=100, type=int, help='Latent vector Size.(Size of generator input)')
ap.add_argument('-ngf', '--map_gen_size', default=64, type=int, help='Size of feature map in Gen ')
ap.add_argument('-ndf', '--map_dis_size', default=64, type=int, help='Size of feature map in Disc ')
dataspecs = [dspec.CELEB]
runner = EasyTorch(ap, dataspecs)

if __name__ == "__main__":
    runner.run(CelebDataset, GANTrainer)
