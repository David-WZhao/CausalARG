import torch.nn as nn
import torch
import numpy as np
from Cross_Attention import CrossAttention
from torch.nn import functional as F
import sys
torch.backends.cudnn.enabled = False
class CML(nn.Module):
    def __init__(self, X_dim, G_dim, z1_dim, z2_dim, transfer_count, mechanism_count, antibiotic_count):
        super(CML, self).__init__()
        self.feature = nn.Sequential(
             # (batch * 1 * 1576 * 23) -> (batch * 32 * 1537 * 20)
             nn.Conv2d(1, 32, kernel_size=(40, 4), ),
             nn.LeakyReLU(),
             # (batch * 32 * 1537 * 20) -> (batch * 32 * 1533 * 19)
             nn.MaxPool2d(kernel_size=(5, 2), stride=1),
             # (batch * 32 * 1533 * 19) -> (batch * 64 * 1504 * 16)
             nn.Conv2d(32, 64, kernel_size=(30, 4)),
             nn.LeakyReLU(),
             # (batch * 64 * 1504 * 16) -> (batch * 128 * 1475 * 13)
             nn.Conv2d(64, 128, kernel_size=(30, 4)),
             nn.LeakyReLU(),
             # (batch * 128 * 1475 * 13) -> (batch * 128 * 1471 * 12)
             nn.MaxPool2d(kernel_size=(5, 2), stride=1),
             # (batch * 128 * 1471, 12) -> (batch * 256 * 1452 * 10)
             nn.Conv2d(128, 256, kernel_size=(20, 3)),
             nn.LeakyReLU(),
             # (batch * 256 * 1452 * 10) -> (batch * 256 * 1433 * 8)
             nn.Conv2d(256, 256, kernel_size=(20, 3)),
             nn.LeakyReLU(),
             # (batch * 256 * 1433 * 8) -> (batch * 256 * 1430 * 8)
             nn.MaxPool2d(kernel_size=(4, 1), stride=1),
             # (batch * 256 * 1430 * 8) -> (batch * 1 * 1411 * 6)
             nn.Conv2d(256, 1, kernel_size=(20, 3)),
             nn.LeakyReLU(),
             # (batch * 1 * 1411 * 6) -> (batch * 1 * 1410 * 6)
             nn.MaxPool2d(kernel_size=(2, 1), stride=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(8460, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, X_dim),
            nn.LeakyReLU()
        )
        self.vae = VAE(X_dim, 512, G_dim)
        self.hidden = Hidden(X_dim, G_dim, z1_dim, z2_dim)
        self.causal = Causal(X_dim + G_dim, transfer_count, mechanism_count, antibiotic_count)

    def forward(self, seq_map):

        x = self.feature(seq_map)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        G, recon_x, mu, logvar = self.vae(x)
        z1, z2 = self.hidden(x, G)
        hidden_representation = torch.cat((z1, z2), dim=1)
        transfer_pre, mechanism_pre, antibiotic_pre = self.causal(hidden_representation)

        return transfer_pre, mechanism_pre, antibiotic_pre, mu, logvar, recon_x, x


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, G_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.G_mean = nn.Linear(hidden_dim, G_dim)
        self.G_logvar = nn.Linear(hidden_dim, G_dim)
        self.decoder = nn.Sequential(
            nn.Linear(G_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.G_mean(hidden)
        logvar = self.G_logvar(hidden)
        G = self.reparameterize(mu, logvar)
        recon_x = self.decoder(G)
        return G, recon_x, mu, logvar

# Define hidden layer network
class Hidden(nn.Module):
    def __init__(self, X_dim, G_dim, z1_dim, z2_dim):
        super(Hidden, self).__init__()
        self.concat_dim = X_dim + G_dim
        self.hidden1 = nn.Sequential(
            nn.Linear(self.concat_dim, self.concat_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.concat_dim * 2, z1_dim),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(G_dim, G_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(G_dim * 2, z2_dim),
        )

    def forward(self, X, G):
        input = torch.cat((X, G), dim=1)
        z1 = self.hidden1(input)
        z2 = self.hidden2(G)
        return z1, z2

# Define the causal diagram module
class Causal(nn.Module):
    def __init__(self, input_dim, transfer_count, mechanism_count, antibiotic_count):
        super(Causal, self).__init__()
        self.transfer_layer = nn.Linear(input_dim, transfer_count)
        self.softmax = nn.Softmax(dim=1)

        self.mechanism_layer = nn.Linear(input_dim + transfer_count, mechanism_count)
        self.antibiotic_layer = nn.Linear(input_dim + transfer_count + mechanism_count, antibiotic_count)


    def forward(self, input):
        transfer_pre = self.softmax(self.transfer_layer(input))
        mechanism_pre = self.softmax(self.mechanism_layer(torch.cat((input, transfer_pre), dim=1)))
        antibiotic_pre = self.softmax(self.antibiotic_layer(torch.cat((input, transfer_pre, mechanism_pre), dim=1)))

        return transfer_pre, mechanism_pre, antibiotic_pre




#Loss function of VAE
def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD


if __name__ == '__main__':
    batch_size = 10
    seq_map = torch.randn(batch_size, 1, 1576, 23)
    print(seq_map.shape)
    model = CML(64, 64, 64, 64, 2, 6, 15)
    print(model)

    transfer, mech, anti = model.forward(seq_map)
    print(transfer.shape, mech.shape, anti.shape)
    # tensor1 = torch.tensor([[1, 2, 3],[4,5,6]])
    # tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
    # cat = torch.cat((tensor1, tensor2), dim=1)
    # print(cat)




