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
parser.add_argument("--batch_size", type=int, help='batch size', default=16)
parser.add_argument("--lr", type=float, help='learning rate', default=1e-4)
parser.add_argument("--epoch", type=int, help='epoch', default=10)
parser.add_argument("--K", type=int, help='K fold', default=5)
parser.add_argument("--X_dim", type=int, help='dimension of X', default=256)
parser.add_argument("--G_dim", type=int, help='dimension of G', default=256)
parser.add_argument("--z1_dim", type=int, help='dimension of z1', default=256)
parser.add_argument("--z2_dim", type=int, help='dimension of z2', default=256)



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
# data, _ = dataloader.load_n_cross_data(1, 4)
# for d, l1, l2, l3 in data:
#     print(l1, l2, l3)
#     exit()
# print(dataloader.load_n_cross_data(1,1))
# train_val_dataloader = dataloader.get_train_val_dataloader()
# assert K == len(train_val_dataloader)

transfer_count, mechanism_count, antibiotic_count = dataloader.get_data_shape()
# print("transfer_count", transfer_count, "mechanism_count", mechanism_count, "antibiotic_count", antibiotic_count)
# exit()
alpha, beta, yita = 1, 0.2, 0.2

#随机数种子
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

# VAE的损失函数
def ELBO(recon_x, x, mu, logvar, anneal=0.025):
    loss_MSE = torch.nn.MSELoss(reduction='mean')
    mse = loss_MSE(recon_x, x)

    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    # print(mse + anneal * KLD)
    return mse + anneal * KLD
    # return anneal * KLD


# Train  初始化各项评估指标
t_transfer_acc, t_transfer_precision, t_transfer_recall, t_transfer_f1 = 0, 0, 0, 0
t_antibiotic_acc, t_antibiotic_precision, t_antibiotic_recall, t_antibiotic_f1 = 0, 0, 0, 0
t_mechanism_acc, t_mechanism_precision, t_mechanism_recall, t_mechanism_f1 = 0, 0, 0, 0

test_dataloader = dataloader.load_test_dataSet(batch_size)

#循环kcross
for k in range(K):
    print('Cross ', k + 1, ' of ', K)
    #模型创建
    model = CML(X_dim, G_dim, z1_dim, z2_dim, transfer_count, mechanism_count, antibiotic_count)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    transfer_loss_function = nn.NLLLoss()
    antibiotic_loss_function = nn.NLLLoss()
    mechanism_loss_function = nn.NLLLoss()


    # train_dataloader = train_val_dataloader[k]['train']
    # val_dataloader = train_val_dataloader[k]['val']
    #加载数据集
    train_dataloader, val_dataloader = dataloader.load_n_cross_data(k + 1, batch_size)

    running_loss = 0.0
    for e in range(epoch):
        df = pd.DataFrame()
        model.train()
        print('train batch: ', len(train_dataloader))
        for index, (seq_map, transfer_label, mechanism_label, antibiotic_label) in enumerate(train_dataloader):
            seq_map, transfer_label, mechanism_label, antibiotic_label = seq_map.view(-1, 1, 1576, 23).to(device), transfer_label.to(device), mechanism_label.to(device), antibiotic_label.to(device)
            optimizer.zero_grad()
            transfer_output, mechanism_output, antibiotic_output, mu, logvar, recon_x, x = model.forward(seq_map)
            # transfer_output, mechanism_output, antibiotic_output= model.forward(seq_map)
            #各模块损失函数
            loss_transfer = transfer_loss_function(torch.log(transfer_output + 0.000001), transfer_label)
            loss_mechanism = mechanism_loss_function(torch.log(mechanism_output + 0.000001), mechanism_label)
            loss_antibiotic = antibiotic_loss_function(torch.log(antibiotic_output + 0.000001), antibiotic_label)

            loss_ELBO = ELBO(recon_x, x, mu, logvar, anneal=0.025)
            # loss = loss_function(torch.log(output), r.long())
            loss = alpha * loss_antibiotic + beta * loss_mechanism + yita * loss_transfer + 0.02 * loss_ELBO
            # loss = alpha * loss_antibiotic + beta * loss_mechanism + yita * loss_transfer

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            df = df._append({'loss_transfer': loss_transfer.item(), 'loss_antibiotic': loss_antibiotic.item(), 'loss_mechanism': loss_mechanism.item(), 'loss': loss.item(), 'running_loss': running_loss}, ignore_index=True)
            if index % 50 == 49:
                # exit()
                print('[%d, %2d, %5d] loss: %.3f' % (k + 1, e + 1, index + 1, running_loss / 50))
                running_loss = 0.0



        df.to_csv('./res/loss_cross' + str(k + 1) + '_epoch' + str(e) + '.csv')
        model.eval()
        val_transfer_pred, val_transfer_label = np.empty(shape=[0, transfer_count]), np.array([])
        val_mechanism_pred, val_mechanism_label = np.empty(shape=[0, mechanism_count]), np.array([])
        val_antibiotic_pred, val_antibiotic_label = np.empty(shape=[0, antibiotic_count]), np.array([])

        for index, (seq_map, transfer_label, mechanism_label, antibiotic_label) in enumerate(val_dataloader):
            seq_map, transfer_label, mechanism_label, antibiotic_label = seq_map.view(-1, 1, 1576, 23).to(device), transfer_label.to(device), mechanism_label.to(device), antibiotic_label.to(device)
            transfer_output, mechanism_output, antibiotic_output,  mu, logvar, recon_x, x = model.forward(seq_map)
            # transfer_output, mechanism_output, antibiotic_output = model.forward(seq_map)

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

    # 测试
    model.eval()
    test_transfer_pred, test_transfer_label = np.empty(shape=[0, transfer_count]), np.array([])
    test_antibiotic_pred, test_antibiotic_label = np.empty(shape=[0, antibiotic_count]), np.array([])
    test_mechanism_pred, test_mechanism_label = np.empty(shape=[0, mechanism_count]), np.array([])
    for index, (seq_map, transfer_label, mechanism_label, antibiotic_label) in enumerate(test_dataloader):
        seq_map, transfer_label, mechanism_label, antibiotic_label = seq_map.view(-1, 1, 1576, 23).to(device), transfer_label.to(device), mechanism_label.to(device), antibiotic_label.to(device)
        transfer_output, mechanism_output, antibiotic_output,  mu, logvar, recon_x, x = model.forward(seq_map)
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

#计算评估指标
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

