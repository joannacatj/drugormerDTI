import pandas as pd
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time
from model import *
from Modules import *
from sklearn.manifold import TSNE
import timeit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
#device=torch.device('cuda')
#将数据导入
def load_tensor(file_name, dtype):
    result=[]
    np.load.__defaults__=(None, True, True, 'ASCII')
    for d in np.load(file_name):
        #print(d)
        #temp=dtype(d).to(device)
        result.append(d)
    np.load.__defaults__=(None, False, True, 'ASCII')
    return result
#打乱数据集
def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

#将数据集拆分为训练集和测试集
def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    DATASET='Human,C.elegans'

    """Load preprocessed data."""

    #dir_input = ('/mnt/sdb/home/hjy/exp1/dataset/' + DATASET+'/Human,C.elegans')
    dir_input='G:/序列预测相关/模型/DrugormerDTI/dataset/Davis/'
    drug = load_tensor(dir_input + 'atom_feat.npy', torch.FloatTensor)
    bond_adj = load_tensor(dir_input + 'bond_adj.npy', torch.FloatTensor)
    dist_adj = load_tensor(dir_input+ 'dist_adj.npy', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins.npy', torch.FloatTensor)
    interactions = load_tensor(dir_input + 'interactions.npy', torch.LongTensor)
    """Create a dataset and split it into train/dev/test."""

    dataset = list(zip(drug, bond_adj, dist_adj, proteins, interactions))
    dataset = shuffle_dataset(dataset,1234)
    dataset_train, dataset_dev = split_dataset(dataset, 0.8)
    print('dataset created!')
    """ create model ,trainer and tester """
    protein_dim = 100#蛋白质的输入维度
    atom_dim = 128
    hid_dim = 128
    n_layers = 3
    n_heads = 8
    pf_dim =256
    dropout = 0.1
    batch = 4
    lr = 1e-4
    weight_decay = 1e-4
    decay_interval = 10
    lr_decay = 1.0
    iteration = 100
    kernel_size = 7

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    graphormer =DrugEncoder(hid_dim)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, MultiHeadAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder,graphormer,decoder,device,atom_dim)
    encoder=encoder.cuda()
    graphormer=graphormer.cuda()
    decoder=decoder.cuda()
    model=model.cuda()
    model.load_state_dict(torch.load("G:\序列预测相关\模型\DrugormerDTI\Davis"))
    #model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)

    """Output files."""
    file_AUCs = 'G:\序列预测相关\模型\DrugormerDTI\Davis6.txt'
    file_model = 'G:\序列预测相关\模型\DrugormerDTI\Davis'
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev\tRecall_dev\tACC_dev\tPrecision_dev\tF1_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    max_AUC_dev = 0
    for epoch in range(1, iteration+1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        loss_train = trainer.train(dataset_train, device)
        #print(loss_train)
        print('Testing……')
        AUC_dev, PRC_dev,recall_dev,accuracy_dev,precision_dev,F1_dev,labels,embeddings,attention = tester.test(dataset_dev)
        end = timeit.default_timer()
        time = end - start
        AUCs = [epoch, time, loss_train, AUC_dev,PRC_dev,recall_dev,accuracy_dev,precision_dev,F1_dev]
        tester.save_AUCs(AUCs, file_AUCs)
        if AUC_dev > max_AUC_dev:
            tester.save_model(model, file_model)
            max_AUC_dev = AUC_dev
        print('\t'.join(map(str, AUCs)))
        print('saving……')
        #print(labels)
        #print(attention.shape)
        emd=[]
        for i in range(len(embeddings)):
            emd.append(embeddings[i][0])
        emd_df=pd.DataFrame(emd)
        emd_df.to_csv(r'G:\序列预测相关\模型\DrugormerDTI\file\embeddings_davis.csv', encoding='utf_8_sig')  # 防止中文乱码
        '''
       #print(attention[-1])
        
        attentions= pd.DataFrame()
        for i in range(len(attention)):
            #print(attention[i].shape)
            if attention[i].shape[1]<300:
                attentions=pd.DataFrame(attention[i].cpu())
            #attentions.append(attention[i][-1].cpu())
        #attention_df = pd.DataFrame(attentions)
        attentions.to_csv('/mnt/sdb/home/hjy/exp1/file/attention.csv', encoding='utf_8_sig')  # 防止中文乱码
       
        file = open('/mnt/sdb/home/hjy/exp1/file/label.txt', 'w', encoding='utf-8')
        for i in range(len(labels)):
            #print(len(labels[i]))
            file.write(str(labels[i]) + '\n')
        file.close()
        file = open('/mnt/sdb/home/hjy/exp1/file/embeddings.txt', 'w', encoding='utf-8')
        for i in range(len(embeddings)):
            print(embeddings[i])
            print("===============================")
            file.write(str(embeddings[i]) + '\n')
        file.close()
        '''

