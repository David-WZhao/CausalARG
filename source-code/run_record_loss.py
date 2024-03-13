import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
import torch.optim as optim
import torch.nn as nn
from argparse import ArgumentParser
from data_loader import ARGDataLoader
from modules import CML
torch.backends.cudnn.enabled = False
from utils import evaluate
import warnings
import random
from random import randint
warnings.filterwarnings("ignore")


parser = ArgumentParser("CML")
# runtime args
parser.add_argument("--device", type=str, help='cpu or gpu', default="cpu")
parser.add_argument("--train_rate", type=float, help='train rate', default=0.8)
parser.add_argument("--batch_size", type=int, help='batch size', default=2)
parser.add_argument("--lr", type=float, help='learning rate', default=1e-4)
parser.add_argument("--epoch", type=int, help='epoch', default=10)
parser.add_argument("--K", type=int, help='K fold', default=1)
parser.add_argument("--X_dim", type=int, help='dimension of X', default=64)
parser.add_argument("--G_dim", type=int, help='dimension of G', default=16)
parser.add_argument("--z1_dim", type=int, help='dimension of z1', default=64)
parser.add_argument("--z2_dim", type=int, help='dimension of z2', default=64)



args = parser.parse_args()

#设置好参数
device = args.device
if args.device != 'cpu':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

train_rate = args.train_rate
batch_size = args.batch_size
lr = args.lr
epoch = args.epoch
K = args.K
X_dim, G_dim = args.X_dim, args.G_dim
z1_dim, z2_dim = args.z1_dim, args.z2_dim
print(str(args))
dataloader = ARGDataLoader()

transfer_count, mechanism_count, antibiotic_count = dataloader.get_data_shape()

alpha, beta, yita, tao= 1, 0.2, 0.2, 0.2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_num = randint(1,1000)
# seed_num = 584
setup_seed(seed_num)
print(seed_num)
# setup_seed(42)



def gauss_loss(mean_logvar1, mean_logvar2, mean_logvar3, mean_logvar4 ,
               batch_mean1, batch_mean2, batch_mean3, batch_mean4):
    loss1 = torch.sum((mean_logvar1 - batch_mean1) ** 2, dim=0)
    loss2 = torch.sum((mean_logvar2 - batch_mean2) ** 2, dim=0)
    loss3 = torch.sum((mean_logvar3 - batch_mean3) ** 2, dim=0)
    loss4 = torch.sum((mean_logvar4 - batch_mean4) ** 2, dim=0)

    loss = torch.sum(loss1) + torch.sum(loss2) + torch.sum(loss3) + torch.sum(loss4)
    return loss

# Add constraint for mean differences
def add_mean_constraint(batch_mean_list, lambda_value):
    mean_diff_loss = 0.0
    num_distributions = len(batch_mean_list)

    # Calculate mean differences
    for i in range(num_distributions - 1):
        for j in range(i + 1, num_distributions):
            mean_diff_loss -= lambda_value * torch.sum((10 * (batch_mean_list[i] - batch_mean_list[j])) ** 2)

    return mean_diff_loss


t_transfer_acc, t_transfer_precision, t_transfer_recall, t_transfer_f1 = 0, 0, 0, 0
t_antibiotic_acc, t_antibiotic_precision, t_antibiotic_recall, t_antibiotic_f1 = 0, 0, 0, 0
t_mechanism_acc, t_mechanism_precision, t_mechanism_recall, t_mechanism_f1 = 0, 0, 0, 0

test_dataloader = dataloader.load_test_dataSet(batch_size)

for k in range(K):
    print('Cross ', k + 1, ' of ', K)

    model = CML(X_dim, G_dim, z1_dim, z2_dim, transfer_count, mechanism_count, antibiotic_count)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    transfer_loss_function = nn.NLLLoss()
    antibiotic_loss_function = nn.NLLLoss()
    mechanism_loss_function = nn.NLLLoss()


    # train_dataloader = train_val_dataloader[k]['train']
    # val_dataloader = train_val_dataloader[k]['val']

    train_dataloader, val_dataloader = dataloader.load_n_cross_data(k + 1, batch_size)

    running_loss = 0.0
    for e in range(epoch):
        mean1_list = []
        mean2_list = []
        mean3_list = []
        mean4_list = []
        prob_list = []
        df = pd.DataFrame()
        model.train()
        print('train batch: ', len(train_dataloader))
        for index, (seq_map, transfer_label, mechanism_label, antibiotic_label) in enumerate(train_dataloader):


            seq_map, transfer_label, mechanism_label, antibiotic_label = seq_map.view(-1, 1, 1576, 23).to(device), \
                transfer_label.to(device), mechanism_label.to(device), antibiotic_label.to(device)


            optimizer.zero_grad()
            transfer_output, mechanism_output, antibiotic_output, mean_logvar1, mean_logvar2, mean_logvar3,mean_logvar4, prob = model.forward(seq_map, transfer_label.view(-1, 1), mechanism_label.view(-1, 1), antibiotic_label.view(-1, 1))

            # print(transfer_output, mechanism_output, antibiotic_output, mean_logvar1, mean_logvar2, mean_logvar3, prob)
            # exit()
            # transfer_output, mechanism_output, antibiotic_output, mu, logvar, recon_x, x = model.forward(seq_map)
            # transfer_output, mechanism_output, antibiotic_output= model.forward(seq_map)
            loss_transfer = transfer_loss_function(torch.log(transfer_output + 0.000001), transfer_label)
            loss_mechanism = mechanism_loss_function(torch.log(mechanism_output + 0.000001), mechanism_label)
            loss_antibiotic = antibiotic_loss_function(torch.log(antibiotic_output + 0.000001), antibiotic_label)



            batch_mean1 = (torch.sum(prob[:, 0, None] * mean_logvar1, dim=0)) / torch.sum(prob[:, 0], dim=0)
            mean1_list.append(batch_mean1)

            # batch_mean2 = torch.mean(outputs[1], dim=0)
            batch_mean2 = (torch.sum(prob[:, 1, None] * mean_logvar2, dim=0)) / torch.sum(prob[:, 1], dim=0)
            mean2_list.append(batch_mean2)

            # batch_mean3 = torch.mean(outputs[2], dim=0)
            batch_mean3 = (torch.sum(prob[:, 2, None] * mean_logvar3, dim=0)) / torch.sum(prob[:, 2], dim=0)
            mean3_list.append(batch_mean3)

            # batch_mean4 = torch.mean(outputs[2], dim=0)
            batch_mean4 = (torch.sum(prob[:, 3, None] * mean_logvar4, dim=0)) / torch.sum(prob[:, 3], dim=0)
            mean4_list.append(batch_mean4)

            batch_prob = torch.mean(prob, dim=0)
            prob_list.append(batch_prob)

            batch_mean_list = [batch_mean1[:len(batch_mean1)], batch_mean2[:len(batch_mean2)],
                               batch_mean3[:len(batch_mean3)], batch_mean4[:len(batch_mean4)]]

            loss_gauss = gauss_loss(mean_logvar1, mean_logvar2, mean_logvar3, mean_logvar4,
                                    batch_mean1, batch_mean2, batch_mean3, batch_mean4)


            # loss_ELBO = ELBO(recon_x, x, mu, logvar, anneal=0.025)
            # loss = loss_function(torch.log(output), r.long())
            loss = alpha * loss_antibiotic + beta * loss_mechanism + yita * loss_transfer + tao * loss_gauss
            +add_mean_constraint(batch_mean_list, omiga)
            # loss = alpha * loss_antibiotic + beta * loss_mechanism + yita * loss_transfer

            # print(add_mean_constraint(batch_mean_list, 0.2))
            # exit()

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()



            df = df._append({'loss_transfer': loss_transfer.item(), 'loss_antibiotic': loss_antibiotic.item(), 'loss_mechanism': loss_mechanism.item(), 'loss': loss.item(), 'running_loss': running_loss}, ignore_index=True)
            if index % 50 == 49:
                # exit()
                print('[%d, %2d, %5d] loss: %.3f' % (k + 1, e + 1, index + 1, running_loss / 50))
                print("gmm_loss",loss_gauss)
                running_loss = 0.0



        df.to_csv('./res/loss_cross' + str(k + 1) + '_epoch' + str(e) + '.csv')
        model.eval()
        val_transfer_pred, val_transfer_label = np.empty(shape=[0, transfer_count]), np.array([])
        val_mechanism_pred, val_mechanism_label = np.empty(shape=[0, mechanism_count]), np.array([])
        val_antibiotic_pred, val_antibiotic_label = np.empty(shape=[0, antibiotic_count]), np.array([])

        for index, (seq_map, transfer_label, mechanism_label, antibiotic_label) in enumerate(val_dataloader):
            seq_map, transfer_label, mechanism_label, antibiotic_label = seq_map.view(-1, 1, 1576, 23).to(device), transfer_label.to(device), mechanism_label.to(device), antibiotic_label.to(device)
            transfer_output, mechanism_output, antibiotic_output, mean_logvar1, mean_logvar2, \
                mean_logvar3, mean_logvar4, prob = model.forward(seq_map,
                torch.full((transfer_label.view(-1, 1).shape), -1).to(device),
                torch.full((transfer_label.view(-1, 1).shape), -1).to(device),
                torch.full((transfer_label.view(-1, 1).shape), -1).to(device))

            transfer_output, transfer_label = transfer_output.cpu().detach().numpy(), transfer_label.cpu().numpy()
            val_transfer_pred = np.append(val_transfer_pred, transfer_output, axis=0)
            val_transfer_label = np.concatenate((val_transfer_label, transfer_label))

            antibiotic_output, antibiotic_label = antibiotic_output.cpu().detach().numpy(), antibiotic_label.cpu().numpy()
            val_antibiotic_pred = np.append(val_antibiotic_pred, antibiotic_output, axis=0)
            val_antibiotic_label = np.concatenate((val_antibiotic_label, antibiotic_label))

            mechanism_output, mechanism_label = mechanism_output.cpu().detach().numpy(), mechanism_label.cpu().numpy()
            val_mechanism_pred = np.append(val_mechanism_pred, mechanism_output, axis=0)
            val_mechanism_label = np.concatenate((val_mechanism_label, mechanism_label))

        print('-------------Val: epoch ' + str(e + 1) + '-----------------')
        acc, macro_p, macro_r, macro_f1 = evaluate(val_transfer_pred, val_transfer_label, transfer_count)
        print('transfer -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
        acc, macro_p, macro_r, macro_f1 = evaluate(val_mechanism_pred, val_mechanism_label, mechanism_count)
        print('mechanism -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
        acc, macro_p, macro_r, macro_f1 = evaluate(val_antibiotic_pred, val_antibiotic_label, antibiotic_count)
        print('antibiotic -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
        print("这一段各个分布的概率是:",prob)
        print("分布1的均值是:",batch_mean1)
        print("分布2的均值是:",batch_mean2)
        print("分布3的均值是:",batch_mean3)
        print("分布4的均值是:",batch_mean4)

    model.eval()
    test_transfer_pred, test_transfer_label = np.empty(shape=[0, transfer_count]), np.array([])
    test_antibiotic_pred, test_antibiotic_label = np.empty(shape=[0, antibiotic_count]), np.array([])
    test_mechanism_pred, test_mechanism_label = np.empty(shape=[0, mechanism_count]), np.array([])
    for index, (seq_map, transfer_label, mechanism_label, antibiotic_label) in enumerate(test_dataloader):
        seq_map, transfer_label, mechanism_label, antibiotic_label = seq_map.view(-1, 1, 1576, 23).to(device), transfer_label.to(device), mechanism_label.to(device), antibiotic_label.to(device)

        transfer_output, mechanism_output, antibiotic_output, mean_logvar1, mean_logvar2, \
            mean_logvar3, mean_logvar4, prob = model.forward(seq_map,
            torch.full((transfer_label.view(-1, 1).shape), -1).to(device),
            torch.full((transfer_label.view(-1, 1).shape), -1).to(device),
            torch.full((transfer_label.view(-1, 1).shape), -1).to(device))

        # transfer_output, mechanism_output, antibiotic_output,  mu, logvar, recon_x, x = model.forward(seq_map)
        # transfer_output, mechanism_output, antibiotic_output = model.forward(seq_map)

        transfer_output, transfer_label = transfer_output.cpu().detach().numpy(), transfer_label.cpu().numpy()
        test_transfer_pred = np.append(test_transfer_pred, transfer_output, axis=0)
        test_transfer_label = np.concatenate((test_transfer_label, transfer_label))

        antibiotic_output, antibiotic_label = antibiotic_output.cpu().detach().numpy(), antibiotic_label.cpu().numpy()
        test_antibiotic_pred = np.append(test_antibiotic_pred, antibiotic_output, axis=0)
        test_antibiotic_label = np.concatenate((test_antibiotic_label, antibiotic_label))

        mechanism_output, mechanism_label = mechanism_output.cpu().detach().numpy(), mechanism_label.cpu().numpy()
        test_mechanism_pred = np.append(test_mechanism_pred, mechanism_output, axis=0)
        test_mechanism_label = np.concatenate((test_mechanism_label, mechanism_label))

    print('========Test: Cross ' + str(k + 1) + '===============')
    acc, macro_p, macro_r, macro_f1 = evaluate(test_transfer_pred, test_transfer_label, transfer_count)
    print('transfer -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
    t_transfer_acc += acc
    t_transfer_precision += macro_p
    t_transfer_recall += macro_r
    t_transfer_f1 += macro_f1

    acc, macro_p, macro_r, macro_f1 = evaluate(test_mechanism_pred, test_mechanism_label, mechanism_count)
    print('mechanism -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
    t_mechanism_acc += acc
    t_mechanism_precision += macro_p
    t_mechanism_recall += macro_r
    t_mechanism_f1 += macro_f1

    acc, macro_p, macro_r, macro_f1 = evaluate(test_antibiotic_pred, test_antibiotic_label, antibiotic_count)
    print('antibiotic -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
    t_antibiotic_acc += acc
    t_antibiotic_precision += macro_p
    t_antibiotic_recall += macro_r
    t_antibiotic_f1 += macro_f1



    torch.save(model.state_dict(), './res/model{}.pth'.format(k))
    # torch.save(model,'./res/modeltotal.pth')
print('transfer => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_transfer_acc / K, t_transfer_precision / K,
                                                                                                 t_transfer_recall / K,
                                                                                                 t_transfer_f1 / K))
print('mechanism => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_mechanism_acc / K, t_mechanism_precision / K,
                                                                                                 t_mechanism_recall / K,
                                                                                                 t_mechanism_f1 / K))
print('antibiotic => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_antibiotic_acc / K, t_antibiotic_precision / K,
                                                                                                 t_antibiotic_recall / K,
                                                                                                 t_antibiotic_f1 / K))
# 将结果写入到文本文件中
with open('./res/result.txt', 'a', encoding='utf8') as f:
    f.write(str(args))
    f.write('\n seed =>{}\n'.format(seed_num))
    f.write('\n transfer => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_transfer_acc / K, t_transfer_precision / K,
                                                                                                         t_transfer_recall / K,
                                                                                                         t_transfer_f1 / K))
    f.write('\n mechanism => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_mechanism_acc / K,
                                                                                                            t_mechanism_precision / K,
                                                                                                            t_mechanism_recall / K,
                                                                                                            t_mechanism_f1 / K))

    f.write('\n antibiotic => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_antibiotic_acc / K,
                                                                                                             t_antibiotic_precision / K,
                                                                                                             t_antibiotic_recall / K,
                                                                                                             t_antibiotic_f1 / K))

    f.write('----------------------------------------------------------------------------------------\n')

# def save_snapshot(model, filename):
#     torch.save(model.state_dict(), filename)
#     f.close()

