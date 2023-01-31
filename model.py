import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from sklearn.metrics import accuracy_score,f1_score
import tqdm
from Radam import *
from lookahead import Lookahead
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
class Predictor(nn.Module):
    def __init__(self, protein_encoder,drug_encoder,decoder, device, atom_dim):
        super().__init__()

        self.protein_encoder = protein_encoder
        self.drug_encoder =drug_encoder
        self.decoder=decoder
        self.device =device
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num)  # batch size
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return compound_mask, protein_mask


    def forward(self,drug,bond_adj,dist_adj,protein,atom_num,protein_num):
        # compound = [batch,atom_num, atom_dim]
        # adj = [batch,atom_num, atom_num]
        # protein = [batch,protein len, 100]
        compound_max_len = drug.shape[1]
        protein_max_len = protein.shape[1]
        #compound_mask, protein_mask = self.make_masks(atom_num, protein_num, compound_max_len, protein_max_len)
        compound=self.drug_encoder(drug,bond_adj,dist_adj)
        #compound = torch.unsqueeze(compound, dim=0)
        # compound = [batch size=1 ,atom_num, atom_dim]
        # protein = torch.unsqueeze(protein, dim=0)
        # protein =[ batch size=1,protein len, protein_dim]
        enc_src = self.protein_encoder(protein)
        # enc_src = [batch size, protein len, hid dim]
        out,emd,attention = self.decoder(compound, enc_src)

        # out = [batch size, 2]
        # out = torch.squeeze(out, dim=0)
        return out,emd,attention

    def __call__(self, data, train=True):

        drug,adj,dist,protein,correct_interaction,atom_num,protein_num = data
        # compound = compound.to(self.device)
        # adj = adj.to(self.device)
        # protein = protein.to(self.device)
        # correct_interaction = correct_interaction.to(self.device)
        Loss = nn.CrossEntropyLoss()

        if train:
            predicted_interaction,_,_ = self.forward(drug,adj,dist,protein,atom_num,protein_num)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss

        else:
            #compound = compound.unsqueeze(0)
            #adj = adj.unsqueeze(0)
            #protein = protein.unsqueeze(0)
            #correct_interaction = correct_interaction.unsqueeze(0)
            predicted_interaction,embedding,attention= self.forward(drug,adj,dist,protein,atom_num,protein_num)
            com_emd=embedding.to('cpu').data.numpy()
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores,com_emd,attention

def pack(atoms,adjs,adj_dist,proteins,labels, device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[1])
        if atom.shape[1] >= atoms_len:
            atoms_len = atom.shape[1]
    protein_num = []

    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    '''
    #print(proteins.shape)
    for protein in proteins:
        protein_num.append(len(protein))
        if len(protein) >= proteins_len:
            proteins_len = len(protein)
    '''
    #atoms_new = torch.zeros((N,7,atom_num), device=device)
    atoms_new = torch.zeros((N, 7, atoms_len), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[1]
        atoms_new[i, :, :a_len] = atom
        i += 1
    #print(atoms_new[0])

    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj=adj.cuda()
        #print(adj.is_cuda)
        adjs_new = torch.zeros((N, a_len, a_len), device=device)
        #print(adjs_new.is_cuda)
        adj = adj + torch.eye(a_len, device=device)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1


    i = 0
    for dist in adj_dist:
        a_len = dist.shape[0]
        dist=dist.cuda()
        adj_dist_new = torch.zeros((N, a_len, a_len), device=device)  # size:atoms_len*atoms_len
        dist = dist + torch.eye(a_len, device=device)
        adj_dist_new[i, :a_len, :a_len] = dist
        i += 1

    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        #protein=torch.from_numpy(protein)
        proteins_new[i, :a_len, :] = protein
        i += 1
    '''

    proteins_new = torch.zeros((N, proteins_len), device=device)
    i = 0
    for protein in proteins:
        a_len = len(protein)
        # print(protein)
        proteins_new[i, :a_len] = protein
        i += 1
    '''
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    return (atoms_new, adjs_new,adj_dist_new,proteins_new, labels_new, atom_num, protein_num)


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.optimizer.zero_grad()
        dists,adjs,atoms,proteins,labels = [],[],[],[],[]
        for data in dataset:
            i = i+1
            atom, adj,dist,protein,label = data
            adjs.append(adj)
            atoms.append(atom)
            dists.append(dist)
            proteins.append(protein)
            labels.append(label)
            if i % self.batch == 0 or i == N:
                data_pack = pack(atoms,adjs,dists,proteins,labels,device)
                loss = self.model(data_pack)
                loss = loss / self.batch
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                dists,adjs, atoms, proteins, labels = [], [], [],[],[]
            else:
                continue
            if i % self.batch == 0 or i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total

# 画ROC曲线函数
def plot_roc_curve(y_true, y_score,figname):
    """
    y_true:真实值
    y_score：预测概率。注意：不要传入预测label！！！
    """
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    fpr,tpr,threshold = roc_curve(y_true, y_score, pos_label=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
    plt.title('Davis roc curve')
    plt.plot(fpr,tpr,color='lightcoral',linewidth=2)
    plt.plot([0,1], [0,1], 'cornflowerblue')
    plt.savefig(fname=figname)

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        #N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            dists, adjs, atoms, proteins, labels = [], [], [], [], []
            attention=[]
            l,com_ebd=[],[]
            for data in dataset:
                atom, adj,dist,protein,label = data
                adjs.append(adj)
                atoms.append(atom)
                dists.append(dist)
                proteins.append(protein)
                labels.append(label)
                data_pack = pack(atoms,adjs,dists,proteins,labels, self.model.device)
                correct_labels, predicted_labels, predicted_scores,compoud_embed,attention_tmp = self.model(data_pack, train=False)
                dists, adjs, atoms, proteins, labels = [], [], [], [], []
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
                l.append(correct_labels)
                com_ebd.append(compoud_embed)
                attention.append(attention_tmp)
                #print(predicted_scores)
                #print(predicted_labels)

        AUC = roc_auc_score(T, S)
        #if AUC>0.8:
        #    plot_roc_curve(T,S,'Human_roc.png')
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        recall=recall_score(T,Y)
        accuracy=accuracy_score(T,Y)
        precision=precision_score(T,Y)
        F1 = f1_score(T, Y, average="micro")
        return AUC, PRC,recall,accuracy,precision,F1,l,com_ebd,attention

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
