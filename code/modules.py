import torch.nn as nn
import torch
import random

torch.backends.cudnn.enabled = False


class CML(nn.Module):
    def __init__(self, X_dim, G_dim, z1_dim, z2_dim, gauus_hidden_dim, transfer_count, mechanism_count,
                 antibiotic_count):
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

        self.gauus = Gauussion(X_dim + 3, G_dim)
        # self.causal = Causal(X_dim+ G_dim, transfer_count, mechanism_count, antibiotic_count)
        # self.latent = latent_layer(X_dim)
        self.hidden = Hidden(X_dim, G_dim, z1_dim, z2_dim)
        self.causal = Causal(z1_dim + z2_dim, transfer_count, mechanism_count, antibiotic_count)

    def forward(self, seq_map, transfer_label, mechanism_label, antibiotic_label):
        x = self.feature(seq_map)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        transfer_label.unsqueeze(-1)
        mechanism_label.unsqueeze(-1)
        antibiotic_label.unsqueeze(-1)

        labels = [transfer_label, mechanism_label, antibiotic_label]
        labels = torch.cat(labels, dim=1)
        # print(labels)
        # mean_logvar1, mean_logvar2, mean_logvar3, prob, G = self.gauus(torch.cat((x, labels), dim=1))
        mean_logvar1, mean_logvar2, mean_logvar3, mean_logvar4, prob, G = self.gauus(torch.cat((x, labels), dim=1))

        z1, z2 = self.hidden(x, G)
        # print(z1, z2)
        # exit()
        hidden_representation = torch.cat((z1, z2), dim=1)

        transfer_pre, mechanism_pre, antibiotic_pre = self.causal(hidden_representation)

        return transfer_pre, mechanism_pre, antibiotic_pre, mean_logvar1, mean_logvar2, mean_logvar3, mean_logvar4, prob


class Hidden(nn.Module):
    def __init__(self, X_dim, G_dim, z1_dim, z2_dim):
        super(Hidden, self).__init__()
        self.concat_dim = X_dim + G_dim
        self.hidden1 = nn.Sequential(
            nn.Linear(self.concat_dim, self.concat_dim),
            nn.LeakyReLU(),
            nn.Linear(self.concat_dim, z1_dim),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(G_dim, G_dim),
            nn.LeakyReLU(),
            nn.Linear(G_dim, z2_dim),
        )

    def forward(self, X, G):
        input = torch.cat((X, G), dim=1)

        z1 = self.hidden1(input)
        z2 = self.hidden2(G)
        return z1, z2


class Causal(nn.Module):
    def __init__(self, input_dim, transfer_count, mechanism_count, antibiotic_count):
        super(Causal, self).__init__()
        self.transfer_layer = nn.Linear(input_dim, transfer_count)
        self.softmax = nn.Softmax(dim=1)

        self.mechanism_layer = nn.Linear(input_dim + transfer_count, mechanism_count)
        self.antibiotic_layer = nn.Linear(input_dim + transfer_count + mechanism_count, antibiotic_count)

        # self.mechanism_layer2 = nn.Linear(input_dim, mechanism_count)
        # self.antibiotic_layer2 = nn.Linear(input_dim, antibiotic_count)

    def forward(self, input):
        transfer_pre = self.softmax(self.transfer_layer(input))
        # mechanism_pre = self.softmax(self.mechanism_layer2(input))
        # antibiotic_pre = self.softmax(self.antibiotic_layer2(input))
        mechanism_pre = self.softmax(self.mechanism_layer(torch.cat((input, transfer_pre), dim=1)))
        antibiotic_pre = self.softmax(self.antibiotic_layer(torch.cat((input, transfer_pre, mechanism_pre), dim=1)))

        return transfer_pre, mechanism_pre, antibiotic_pre


class Gauussion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Gauussion, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.mean_logvar1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.mean_logvar2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.mean_logvar3 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.mean_logvar4 = nn.Linear(hidden_dim, 2 * hidden_dim)

        self.softmax = nn.Softmax(dim=1)
        # self.prob = nn.Linear(hidden_dim, 3)
        self.prob = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        hidden = self.hidden(x)
        mean_logvar1 = self.mean_logvar1(hidden)
        mean_logvar2 = self.mean_logvar2(hidden)
        mean_logvar3 = self.mean_logvar3(hidden)
        mean_logvar4 = self.mean_logvar4(hidden)
        prob = self.softmax(self.prob(hidden))


        values = [0, 1, 2, 3]
        # print(prob)
        # index = []
        g_list = []
        mid = hidden.size()[1]
        for i, pro in enumerate(prob):

            value = random.choices(values, pro.tolist())[0]

            if (value == 0):
                g = mean_logvar1[i][:mid] + torch.rand_like(mean_logvar1[i][mid:]) * mean_logvar1[i][mid:]
            elif (value == 1):
                g = mean_logvar2[i][:mid] + torch.rand_like(mean_logvar2[i][mid:]) * mean_logvar2[i][mid:]
            elif (value == 2):
                g = mean_logvar3[i][:mid] + torch.rand_like(mean_logvar3[i][mid:]) * mean_logvar3[i][mid:]
            else:
                g = mean_logvar4[i][:mid] + torch.rand_like(mean_logvar4[i][mid:]) * mean_logvar4[i][mid:]

            g_list.append(g)


        final_g = torch.stack(g_list)
        return mean_logvar1, mean_logvar2, mean_logvar3, mean_logvar4, prob, final_g


if __name__ == '__main__':
    batch_size = 16
    seq_map = torch.randn(batch_size, 1, 1576, 23)
    # print(seq_map.shape)
    model = CML(5, 2, 5, 5, 5, 2, 6, 15)
    print(model)
    transfer_label = torch.randn(batch_size, 1)
    mechanism_label = torch.randn(batch_size, 1)
    antibiotic_label = torch.randn(batch_size, 1)
    device = 'cpu'
    # transfer, mech, anti,_,_,_,_ = model.forward(seq_map, transfer_label, mechanism_label, antibiotic_label)
    transfer, mech, anti, _, _, _, _ = model.forward(seq_map, torch.zeros(batch_size, 1).to(device),
                                                     torch.zeros(batch_size, 1).to(device),
                                                     torch.zeros(batch_size, 1).to(device))
    # print(transfer.shape, mech.shape, anti.shape)

    # print(torch.zeros(batch_size, 1).to(device))
    # tensor1 = torch.tensor([[1, 2, 3],[4,5,6]])
    # tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
    # cat = torch.cat((tensor1, tensor2), dim=1)
    # print(cat)
