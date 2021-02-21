import os

import torch
import torchvision.transforms as tmf
import torchvision.utils as vutils
from PIL import Image as IMG
from easytorch import EasyTorch, ETTrainer, ETDataset, ETMetrics, ETAverages
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
        return {'input': img}

    @property
    def transforms(self):
        return tmf.Compose(
            [tmf.Resize(self.image_size), tmf.CenterCrop(self.image_size),
             tmf.ToTensor(), tmf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class GANTrainer(ETTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.real_label = 1
        self.fake_label = 0
        self.criterion = torch.nn.BCELoss()
        self.fixed_noise = torch.randn(self.args['batch_size'], self.args['latent_size'], 1, 1)

    def _init_nn_model(self):
        self.nn['gen'] = Generator(self.args['num_channel'], self.args['latent_size'], self.args['map_gen_size'])
        self.nn['dis'] = Discriminator(self.args['num_channel'], self.args['map_dis_size'])

    def _init_optimizer(self):
        self.optimizer['gen'] = torch.optim.Adam(self.nn['gen'].parameters(), lr=self.args['learning_rate'],
                                                 betas=(0.5, 0.999))
        self.optimizer['dis'] = torch.optim.Adam(self.nn['dis'].parameters(), lr=self.args['learning_rate'],
                                                 betas=(0.5, 0.999))

    def training_iteration(self, i, batch):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        self.nn['dis'].zero_grad()
        # Format batch
        real_images = batch['input'].to(self.device['gpu'])
        b_size = real_images.size(0)
        label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device['gpu'])
        # Forward pass real batch through D
        output = self.nn['dis'](real_images).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, self.args['latent_size'], 1, 1, device=self.device['gpu'])
        # Generate fake image batch with G
        fake = self.nn['gen'](noise)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output = self.nn['dis'](fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizer['dis'].step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.nn['gen'].zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.nn['dis'](fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizer['gen'].step()

        losses = self.new_averages()
        losses.add(errD.item(), len(batch['input']), index=0)
        losses.add(errG.item(), len(batch['input']), index=1)
        return {'averages': losses, 'real_images': real_images}

    def _on_iteration_end(self, i, ep, it):
        if i % 500 == 0:  # Save every 500th multiple batch
            fake = self.nn['gen'](self.fixed_noise.to(self.device['gpu'])).detach().cpu()
            grid = vutils.make_grid(fake, padding=2, normalize=True)
            vutils.save_image(grid, f"{self.cache['log_dir']}{sep}{i}_fake.png")

            grid = vutils.make_grid(it['real_images'], padding=2, normalize=True)
            vutils.save_image(grid, f"{self.cache['log_dir']}{sep}{i}_real.png")

    def new_averages(self):
        return ETAverages(num_averages=2)

    def init_experiment_cache(self):
        self.cache['log_header'] = 'D_Loss,G_Loss'


CELEB = {
    'name': 'CELEB',
    'data_dir': 'img_align_celeba'
}
runner = EasyTorch([CELEB], phase='train', dataset_dir='datasets',
                   learning_rate=0.0002, split_ratio=[1], epochs=15, num_channel=3,
                   latent_size=100, map_gen_size=64, map_dis_size=64)

if __name__ == "__main__":
    runner.run(GANTrainer, CelebDataset)
