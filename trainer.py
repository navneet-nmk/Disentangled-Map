from variational_autoencoder import *
import torch
import torch.optim as optim
import gym
import gym_minigrid
from env_wrappers import FlatObsWrapper
import numpy as np
import loss
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io, transform

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)


class StatesDataset(Dataset):
    def __init__(self, data, new_h=8, new_w=8, transform=None):
        super(StatesDataset, self).__init__()
        self.data = data
        self.transform = transform
        self.new_h = new_h
        self.new_w = new_w

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        state_true_1 = self.data[item]
        state_true_2 = self.data[np.random.randint(low=0, high=len(self.data))]
        state_true_1 = transform.resize(state_true_1, (self.new_h, self.new_w))
        state_true_2 = transform.resize(state_true_2, (self.new_h, self.new_w))
        state_true_1 = np.float32(state_true_1)
        state_true_2 = np.float32(state_true_2)
        if self.transform:
            state_true_1 = self.transform(state_true_1)
            state_true_2 = self.transform(state_true_2)
        return state_true_1, state_true_2


class Trainer(object):

    def __init__(self,
                 autoencoder,
                 discriminator,
                 lr_vae,
                 lr_disc,
                 num_epochs,
                 environment,
                 beta,
                 gamma,
                 batch_size,
                 num_samples,
                 model_save_dir,
                 num_workers=4
                 ):

        self.model = autoencoder
        self.disc = discriminator
        self.optim_vae = optim.Adam(lr=lr_vae, params=self.model.parameters())
        self.optim_disc = optim.Adam(lr=lr_disc, params=self.disc.parameters())
        self.num_epochs = num_epochs
        self.env = environment
        self.gamma = gamma
        self.beta = beta
        self.writer = SummaryWriter()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_workers = num_workers
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.dataloader = self.create_dataloader()
        self.ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        self.zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.model.to(self.device)
        self.disc.to(self.device)
        self.model_save_dir = model_save_dir

    def collect_data(self):
        data = []
        state = self.env.reset()
        data.append(state['image'])
        for _ in range(1000):
            next_state, reward, done, _ = self.env.step(self.env.action_space.sample())  # take a random action
            data.append(next_state['image'])
        return data

    def create_dataloader(self):
        states_dataset = StatesDataset(self.collect_data(), transform=transforms.Compose(
            [transforms.ToTensor()]))
        self.dataloader = DataLoader(states_dataset, self.batch_size, num_workers=self.num_workers)
        return self.dataloader

    def train(self):
        # Create the dataloader
        self.dataloader = self.create_dataloader()
        for e in range(self.num_epochs):
            discriminator_loss = 0
            va_loss = 0
            for i, states_batch in enumerate(self.dataloader):

                states_true_1, states_true_2 = states_batch

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
                try:
                    D_tc_loss = 0.5 * (F.cross_entropy(d_z, self.zeros) + F.cross_entropy(D_z_pperm, self.ones))
                except:
                    batch_size, _ = d_z.shape
                    ones = torch.ones(batch_size, dtype=torch.long, device=self.device)
                    zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                    D_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                self.optim_disc.zero_grad()
                D_tc_loss.backward()
                self.optim_disc.step()

                discriminator_loss += D_tc_loss.item()

            # Add the loss to tensorboard
            self.writer.add_scalar('data/disc_loss', discriminator_loss/len(self.dataloader), e)
            self.writer.add_scalar('data/vae_loss', va_loss / len(self.dataloader), e)

        self.writer.close()

    def save_model(self):
        print("Saving the model at ", self.model_save_dir+'model.pt')
        torch.save(self.model, self.model_save_dir)


if __name__ == '__main__':
    # Create the environment
    env = gym.make('MiniGrid-DoorKey-16x16-v0')

    observation_space = env.observation_space
    action_space = env.action_space
    img_pace = env.observation_space.spaces['image']

    # Model Variables
    input_channels = 3
    conv_layers = 64
    height = 8
    width = 8
    latent_dim = 128
    batch_size = 8
    lr_vae = 1e-2
    lr_disc = 1e-2

    model_save_dir = '/Users/navneetmadhukumar/PycharmProjects/disentangled-minigrid/'

    model = VAE(conv_layers=conv_layers,
                height=height,
                width=width,
                input_channels=input_channels,
                latent_dim=latent_dim)

    discriminator = Discriminator(latent_dim=latent_dim, hidden_dim=1000, output_dim=2)

    trainer = Trainer(model, discriminator, lr_vae=lr_vae,
                      lr_disc=lr_disc, batch_size=batch_size,
                      beta=10, environment=env, gamma=1, num_epochs=50,
                      num_samples=1000, model_save_dir=model_save_dir)

    trainer.train()



