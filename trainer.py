from variational_autoencoder import *
import torch
import torch.optim as optim
import gym
import numpy as np
import loss
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)

class StatesDataset(Dataset):
    def __init__(self, data):
        super(StatesDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        state_true_1 = self.data[item]
        state_true_2 = self.data[np.random.randint(low=0, high=len(self.data))]
        return state_true_1, state_true_2


class Trainer(object):

    def __init__(self,
                 autoencoder,
                 discriminator,
                 optim_vae,
                 optim_disc,
                 num_epochs,
                 environment,
                 dataloader,
                 beta,
                 gamma,
                 batch_size,
                 num_samples,
                 num_workers=4
                 ):

        self.model = autoencoder
        self.disc = discriminator
        self.optim_vae = optim_vae
        self.optim_disc = optim_disc
        self.num_epochs = num_epochs
        self.env = environment
        self.dataloader = dataloader
        self.gamma = gamma
        self.beta = beta
        self.writer = SummaryWriter()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_workers = num_workers

        self.ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        self.zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'

    def collect_data(self):
        data = []
        state = self.env.reset()
        data.append(state)
        for _ in range(1000):
            next_state, reward, done, _ = self.env.step(self.env.action_space.sample())  # take a random action
            data.append(next_state)
        return data

    def create_dataloader(self):
        states_dataset = StatesDataset(self.collect_data())
        self.dataloader = DataLoader(states_dataset, self.batch_size, num_workers=self.num_workers)
        return self.dataloader

    def train(self):
        for e in range(self.num_epochs):
            discriminator_loss = 0
            va_loss = 0
            for i, states_true_1, states_true_2 in enumerate(self.dataloader):

                states_true_1 = states_true_1.to(self.device)
                states_true_2 = states_true_2.to(self.device)

                reconstruction, latent, mu, logvar = self.model(states_true_1)
                vae_recon_loss = loss.recon_loss(states_true_1, reconstruction)
                vae_kl_divergence = loss.kl_divergence(mu, logvar)
                d_z = self.disc(latent)
                vae_tc_loss = (d_z[:, :1] - d_z[:, 1:]).mean()

                vae_loss = vae_recon_loss + vae_kl_divergence + self.beta * vae_tc_loss

                self.optim_vae.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_vae.step()

                va_loss += vae_loss.item()

                states_true_2 = states_true_2.to(self.device)
                z_prime = self.model(states_true_2, no_dec=True)
                z_pperm = loss.permute_dims(z_prime).detach()
                D_z_pperm = self.disc(z_pperm)
                D_tc_loss = 0.5 * (F.cross_entropy(d_z, self.zeros) + F.cross_entropy(D_z_pperm, self.ones))

                self.optim_disc.zero_grad()
                D_tc_loss.backward()
                self.optim_disc.step()

                discriminator_loss += D_tc_loss.item()

            # Add the loss to tensorboard
            self.writer.add_scalar('data/disc_loss', discriminator_loss/len(self.dataloader), e)
            self.writer.add_scalar('data/vae_loss', va_loss / len(self.dataloader), e)

        self.writer.close()