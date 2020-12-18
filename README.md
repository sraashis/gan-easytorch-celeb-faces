# GAN example for easytorch.
### 1. Dataset can be downloaded at [CelebFaces Attributes Datasets](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
### 2. Models/Implementation example for Generator and Discriminator are used from the examples:
- [DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [GANs from scratch](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f)
### How to run?
#### 1. Download/extract dataset from above in gan-easytorch-celeb-faces/datasets/ folder.
#### 2. python main.py python main.py -ph train -rt 1 -b 128 -lr 0.0005
- -ph is phase (train, validation, test)
- -rt is split data to just one (train). If it was (0.8, 0.2), the data would be split in train, test set.
- -b batch size
- -lr learning rate
```python
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
```

```python
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

```
Run
```python
dataspecs = [dspec.CELEB]
runner = EasyTorch(ap, dataspecs)

if __name__ == "__main__":
    runner.run(CelebDataset, GANTrainer)
```