import os

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
#from tape import TAPETokenizer
from rdkit import Chem
import torch
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
BOND_ORDER_MAP = {0: 0, 1: 1, 1.5: 2, 2: 3, 3: 4}
dataset=pd.read_csv(r'G:\序列预测相关\模型\graph-transformer\dataset\KIBA\KIBA.txt',sep = ' ',
                  names =['drug id','protein id','drug','protein','compound'])
#处理protein部分
protein=dataset['protein']
def seq_to_kmers(seq, k=3):
    """ Divide a string into a list of kmers strings.

    Parameters:
        seq (string)
        k (int), default 3
    Returns:
        List containing a list of kmers.
    """
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]
class Corpus(object):
    """ An iteratable for training seq2vec models. """

    def __init__(self,data, ngram):
        self.df = data
        self.ngram = ngram

    def __iter__(self):
        for no, data in enumerate(self.df):
            yield  seq_to_kmers(data,self.ngram)
def get_protein_embedding(model,protein):
    """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    vec = np.zeros((len(protein), 100))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i += 1
    vec=torch.from_numpy(vec)
    return np.array(vec)
sent_corpus = Corpus(protein,3)
model = Word2Vec(vector_size=100, window=5, min_count=0, workers=6)
model.build_vocab(sent_corpus)
model.train(sent_corpus,epochs=30,total_examples=model.corpus_count)
model.save("word2vec_30_DrugBank.model")
#model=Word2Vec.load('word2vec_30_DrugBank.model')
print('model end!')
#model = Word2Vec.load('/mnt/sdb/home/hjy/exp1/word2vec_30_GPCR_train.model')
proteins=[]
for no, data in enumerate(protein):
    protein_embedding = get_protein_embedding(model, seq_to_kmers(data))
    proteins.append(protein_embedding)
print('protein end')

#处理drug部分（把其转换为分子图）
drug=dataset['drug']
drug= drug.values.tolist()
#处理分子信息部分
def get_atoms_info(mol):
    atoms = mol.GetAtoms()
    n_atom = len(atoms)
    atom_fea = torch.zeros(7, n_atom, dtype=torch.half)
    AllChem.ComputeGasteigerCharges(mol)
    for idx, atom in enumerate(atoms):
        atom_fea[0, idx] = atom.GetAtomicNum()
        atom_fea[1, idx] = atom.GetTotalDegree() + 1
        atom_fea[2, idx] = int(atom.GetHybridization()) + 1
        atom_fea[3, idx] = atom.GetTotalNumHs() + 1
        atom_fea[4, idx] = atom.GetIsAromatic() + 1
        for n_ring in range(3, 9):
            if atom.IsInRingSize(n_ring):
                atom_fea[5, idx] = n_ring + 1
                break
        else:
            if atom.IsInRing():
                atom_fea[5, idx] = 10
        atom_fea[6, idx] = atom.GetDoubleProp("_GasteigerCharge")*10

    atom_fea = torch.nan_to_num(atom_fea)
    return np.array(atom_fea), n_atom

#处理图边的信息部分
def get_bond_order_adj(mol):
    n_atom = len(mol.GetAtoms())
    bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.uint8)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_adj[i, j] = bond_adj[j, i] = BOND_ORDER_MAP[bond.GetBondTypeAsDouble()]
    return np.array(bond_adj)
def get_bond_adj(mol):
    """
    :param mol: rdkit mol
    :return: multi graph for {
                sigmoid_bond_graph,
                pi_bond_graph,
                2pi_bond_graph,
                aromic_graph,
                conjugate_graph,
                ring_graph,
    }
    """
    n_atom = len(mol.GetAtoms())
    bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.uint8)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        bond_adj[i, j] = bond_adj[j, i] = 1
        if bond_type in [2, 3]:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 1)
        if bond_type == 3:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 2)
        if bond_type == 1.5:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 3)
        if bond.GetIsConjugated():
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 4)
        if bond.IsInRing():
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 5)
    return np.array(bond_adj)
#处理边的距离部分
def get_dist_adj(mol, use_3d_info=False):
    if use_3d_info:
        m2 = Chem.AddHs(mol)
        is_success = AllChem.EmbedMolecule(m2, enforceChirality=False)
        if is_success == -1:
            dist_adj = None
        else:
            AllChem.MMFFOptimizeMolecule(m2)
            m2 = Chem.RemoveHs(m2)
            dist_adj = (-1 * torch.tensor(AllChem.Get3DDistanceMatrix(m2), dtype=torch.float))
    else:
        dist_adj = (-1 * torch.tensor(molDG.GetMoleculeBoundsMatrix(mol), dtype=torch.float))
    return np.array(dist_adj)
#总的调用函数
def smile_to_mol_info(smile, calc_dist=True, use_3d_info=False):
    mol = Chem.MolFromSmiles(smile)
    bond_adj = get_bond_adj(mol)
    dist_adj = get_dist_adj(mol) if calc_dist else None
    dist_adj_3d = get_dist_adj(mol, use_3d_info) if calc_dist else None
    atom_fea, n_atom = get_atoms_info(mol)
    return atom_fea,bond_adj,dist_adj,n_atom
atoms,adjs,dists,nums=[],[],[],[]
for no, data in enumerate(drug):
    fea,adj,dist,atom_num=smile_to_mol_info(data)
    atoms.append(fea)
    adjs.append(adj)
    dists.append(dist)
    nums.append(atom_num)
print('drug end')
#atoms= torch.tensor([item.cpu().detach().numpy() for item in atoms]).cuda()
interactions=dataset['compound'].values.tolist()
dir_input = 'G:/序列预测相关/模型/DrugormerDTI/dataset/DrugBank/'
os.makedirs(dir_input, exist_ok=True)
np.save(dir_input + 'atom_feat',atoms )
np.save(dir_input + 'bond_adj', adjs)
np.save(dir_input + 'dist_adj',dists)
np.save(dir_input + 'proteins', proteins)
np.save(dir_input + 'interactions', interactions)
print('The preprocess of dataset has finished!')
