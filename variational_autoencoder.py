import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

print("Using torch version ", torch.__version__)


# This variational autoencoder also includes a factor loss term to encourage disentangled representations.
class VAE(nn.Module):

    def __init__(self,
                 input_channels,
                 height, width,
                 conv_layers,
                 latent_dim,
                 kernel_size=3,
                 stride=2,
                 usebatchnorm=False,
                 init_weights=True,
                 padding='same'):
        super(VAE, self).__init__()

        self.height = height
        self.width = width
        self.conv_layers = conv_layers
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.init_weights = init_weights
        self.usebatchnorm = usebatchnorm

        self.pad = (self.kernel_size-1)//2
        self.stride = stride

        # Encode the input state s to a latent variable z -> P(z|s)
        self.conv1 = nn.Conv2d(in_channels=input-input_channels, out_channels=self.conv_layers,
                               kernel_size=self.kernel_size, padding=self.pad, stride=self.stride)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers*2,
                               kernel_size=self.kernel_size, padding=self.pad, stride=self.stride)

        linear_input_shape = (self.height//4)*(self.width//4)*self.conv_layers*2
        self.mu = nn.Linear(in_features=linear_input_shape, out_features=self.latent_dim)
        self.logvar = nn.Linear(in_features=linear_input_shape, out_features=self.latent_dim)

        # Decode the latent dim z to the state s -> P(s|z)
        self.fc1 = nn.Linear(in_features=self.latent_dim, out_features=linear_input_shape)
        self.conv1_dec = nn.ConvTranspose2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                                            stride=self.stride, padding=self.pad, kernel_size=self.kernel_size)
        self.conv2_dec = nn.ConvTranspose2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers,
                                            stride=self.stride, padding=self.pad, kernel_size=self.kernel_size)
        self.output = nn.ConvTranspose2d(in_channels=self.conv_layers, out_channels=self.input_channels,
                                         padding=self.pad, kernel_size=self.kernel_size)

        if self.init_weights:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            nn.init.xavier_uniform_(self.mu.weight)
            nn.init.xavier_uniform_(self.logvar.weight)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.conv1_dec.weight)
            nn.init.xavier_uniform_(self.conv2_dec.weight)
            nn.init.xavier_uniform_(self.output.weight)

    def encode(self, state):
        batch_size, _, _, _ = state.shape
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.reshape(batch_size, -1)
        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick as shown in the auto encoding variational bayes paper
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = F.relu(self.fc1(z))
        z = z.reshape(-1, self.conv_layers*2, self.width//4, self.height//4)
        z = F.relu(self.conv1_dec(z))
        z = F.relu(self.conv2_dec(z))
        output = self.output(z)
        return output

    def forward(self, state):
        mu, logvar = self.encode(state)
        latent = self.reparameterize(mu, logvar)
        reconstructed = self.decode(latent)
        return reconstructed, latent


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


# Discriminator class that is used for the disentangling of the features
class Discriminator(nn.Module):
    def __init__(self,
                 latent_dim,
                 hidden_dim, output_dim, init_weight=True):
        super(Discriminator, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output = output_dim
        self.init_w = init_weight

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_dim, self.output),
        )

        if self.init_w:
            self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z)


    














