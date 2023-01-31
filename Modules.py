import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
import copy
import math
PI = 3.14159
A = (2 * PI) ** 0.5

#这部分是protein的处理部分，就是transformer最基本的encoder,借用了transformer的论文的代码
class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size, dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout).cuda()
        self.fc = nn.Linear(self.input_dim, self.hid_dim).cuda()
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)
        #self.embedding=nn.Embedding(10648,self.hid_dim)

    def forward(self, protein):
        #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        #protein = protein + self.pos_embedding(pos)
        #protein = [batch size, protein len,protein_dim]
        #conv_input=self.embedding(protein.to(torch.int64))
        #print(protein_embedding.Size())
        conv_input = self.fc(protein)
        #print(conv_input.shape)
        #conv_input=[batch size,protein len,hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]

            #pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        # conved = [batch size,protein len,hid dim]
        conved = self.ln(conved)
        #print('protein embedding:',conved.shape)
        return conved
#以下是整个获取embedding的方法！！
class GraphormerEmbeddingLayer(nn.Module):
    def __init__(self, d_model, n_head, max_paths, n_graph_type, max_single_hop, atom_dim, total_degree,
                 hybrid, hydrogen, aromatic, ring, n_layers, need_graph_token,
                 use_3d_info=False, dropout=0.,use_dist_adj=True):
        super(GraphormerEmbeddingLayer, self).__init__()

        self.atom_encoder = AtomFeaEmbedding(d_model, atom_dim, total_degree, hybrid,
                                             hydrogen, aromatic, ring, n_layers, need_graph_token)
        self.edge_encoder = EdgeEmbedding(d_model, n_head, max_paths, n_graph_type, max_single_hop, n_layers,
                                          need_graph_token, use_3d_info=use_3d_info, use_dist_adj=use_dist_adj)

    def forward(self, atom_fea, bond_adj, dist_adj,dist3d_adj=None, contrast=False):
        return self.atom_encoder(atom_fea,contrast), \
               self.edge_encoder(bond_adj, dist_adj, dist3d_adj, contrast)
#1.centrality embedding+atomfeats的表示
#根据数据集的处理，目前知道的有总原子数，总建数，杂化方式，含氢量，是否为芳香烃，几元环等特征，所以init对应其one-hot需要的维度，不够可以调整
class AtomFeaEmbedding(nn.Module):
    def __init__(self, d_model, atom_dim=100, total_degree=10, hybrid=10,
                 hydrogen=8, aromatic=2, ring=10, n_layers=1, need_graph_token=True):
        super(AtomFeaEmbedding, self).__init__()
        self.atom_encoders = nn.ModuleList([
            nn.Embedding(100, d_model, padding_idx=0).cuda(),
            nn.Embedding(total_degree + 1, d_model, padding_idx=0).cuda(),
            nn.Embedding(20, d_model, padding_idx=0).cuda(),
            nn.Embedding(hydrogen + 1, d_model, padding_idx=0).cuda(),
            nn.Embedding(aromatic + 1, d_model, padding_idx=0).cuda(),
            nn.Embedding(ring + 1, d_model, padding_idx=0).cuda(),
            GaussianAtomLayer(d_model, means=(-1, 1), stds=(0.1, 10))
        ])
        self.need_graph_token = need_graph_token
        # self.feed_forward = nn.Sequential(
        #     nn.LayerNorm(d_model),
        # )

        if need_graph_token:
            self.graph_token = nn.Embedding(2, d_model).cuda()
            self.contrast_token = nn.Embedding(2, d_model).cuda()

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self,atom_fea,contrast=False):
        """
        :param atom_fea: [bsz, n_fea_type, n_atom]
        :return: [bsz, n_atom + I, d_model] I = 1 if need_graph_token else 0
        """
        #这里改过，atom_fea格式不一样

        bsz, n_fea_type, n_atom = atom_fea.size()
        #print(atom_fea)
        #print('atom_fea embeddings:',atom_fea.shape)
        #print('n_atom',n_atom)
        out = self.atom_encoders[-1](atom_fea[:, -1])
        #print(self.atom_encoders[2](atom_fea[:, 2].int()).is_cuda)
        #print(self.embedding.num_embeddings)
        for idx in range(6):
            #print(atom_fea[:, idx].int())
            #print(self.atom_encoders[idx].weight.size(), atom_fea[:, idx].max(dim=0))
            #print(out.is_cuda)
            out += self.atom_encoders[idx](atom_fea[:, idx].int())
            #out += self.atom_encoders[idx](atom_fea[:, idx])
            #print(out)
        if self.need_graph_token:
            graph_token = self.graph_token.weight[0].clone()
            if contrast:
                graph_token += self.contrast_token.weight[0]
            graph_token = graph_token.view(1, 1, -1).repeat(bsz, 1, 1)
            out = torch.cat([graph_token, out], dim=1)
        return out
class GaussianAtomLayer(nn.Module):
    def __init__(self, d_model=128, means=(0, 3), stds=(0.1, 10),device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.means = nn.Embedding(1, d_model).cuda()
        self.stds = nn.Embedding(1, d_model).cuda()
        self.mul = nn.Embedding(1, 1).cuda()
        self.bias = nn.Embedding(1, 1).cuda()
        nn.init.uniform_(self.means.weight, means[0], means[1])
        nn.init.uniform_(self.stds.weight, stds[0], stds[1])
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
    def real_forward(self, x):
        mul = self.mul.weight
        bias = self.bias.weight
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, self.d_model)
        mean = self.means.weight[0]
        std = self.stds.weight[0].abs() + 1e-5
        return gaussian(x, mean, std).type_as(self.means.weight)
    def forward(self, x):
        """
        :param x: [bsz, n_atom]
        :return: [bsz, n_atom, d_model]
        """
        out = self.real_forward(x)
        return torch.where(x.unsqueeze(-1).expand_as(out) != 0, out, torch.zeros_like(out))
#2.Spatial Encoding（空间编码）的方法
#对应edge_embedding层中spatial encoder的部分，运用了Gaussian核函数处理方法(可以尝试替换为其他核函数)
def gaussian(x, mean, std):
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (A * std)
class GaussianBondLayer(nn.Module):
    def __init__(self, nhead=16, means=(0, 3), stds=(0.1, 10)):
        super().__init__()
        self.nhead = nhead
        self.means = nn.Embedding(1, nhead).cuda()
        self.stds = nn.Embedding(1, nhead).cuda()
        self.mul = nn.Embedding(1, 1).cuda()
        self.bias = nn.Embedding(1, 1).cuda()
        nn.init.uniform_(self.means.weight, means[0], means[1])
        nn.init.uniform_(self.stds.weight, stds[0], stds[1])
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def real_forward(self, x):
        mul = self.mul.weight
        bias = self.bias.weight
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.nhead)
        mean = self.means.weight[0]
        std = self.stds.weight[0].abs() + 1e-5
        return gaussian(x, mean, std).type_as(self.means.weight)

    def forward(self, x):
        """
        :param x: [bsz, n_atom, n_atom]
        :return: [bsz, n_atom, n_atom, nhead]
        """
        out = self.real_forward(x)
        return torch.where(x.unsqueeze(-1).expand_as(out) != 0, out, torch.zeros_like(out))

#3.Edge Encoding(边编码)的方法
#对应edge_emebedding层中edge encoder的部分
#因为在模型当中edge_encoding和spatial_encoding最后要concat在一起以后喂给transformer,所以这里EdgeEmbedding的输出是把两个结合起来过以后！
class EdgeEmbedding(nn.Module):
    def __init__(self, embed_dim, n_head=16, max_paths=50, n_graph_type=6, max_single_hop=4, n_layers=1,
                 need_graph_token=True, use_3d_info=False, use_dist_adj=True):
        """
        :param embed_dim:
        :param n_head:  n_head mast has to be 2 to the nth power
                        example: n_head=2^4, hop_distribution=[1, 2^3, 2^2, 2^1, 2^0]
                                 n_head=2^n, hop_distribution=[1, 2^(n-1), 2^(n-2), ..., 2^0]
        :param max_paths:
        :param n_graph_type:
        :param max_single_hop:
        :param n_layers:
        """
        super().__init__()
        self.num_heads = n_head
        self.embed_dim = embed_dim
        # assert n_head & (n_head - 1) == 0, f"n_head mast has to be 2 to the nth power, but got{n_head}"
        self.n_graph_type = n_graph_type
        self.max_single_hop = max_single_hop
        self.use_3d_info = use_3d_info
        self.use_dist_adj = use_dist_adj
        # self.hop_distribution = [1 << i for i in range(int(math.log2(n_head) - 1), -1, -1)]
        # self.hop_distribution[-1] += 1
        # assert sum(self.hop_distribution) == self.num_heads

        self.head_dim = embed_dim // n_head
        assert self.head_dim * n_head == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.edge_encoders = nn.ModuleList([
            nn.Embedding(max_paths + 1, n_head, padding_idx=0).cuda() for _ in range(n_graph_type)
        ])
        self.spatial_encoder = GaussianBondLayer(n_head, means=(0, 3), stds=(0.1, 10))
        if self.use_3d_info:
            self.spatial3d_encoder = GaussianBondLayer(n_head, means=(0, 3), stds=(0.1, 10))

        # self.norm = nn.LayerNorm(n_head)

        self.need_graph_token = need_graph_token
        if need_graph_token:
            self.graph_token = nn.Embedding(1, n_head).cuda()
            self.contrast_token = nn.Embedding(1, n_head).cuda()

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, bond_adj, dist_adj, dist3d_adj=None, contrast=False):
        """
        :param bond_adj: [bsz, n_atom, n_atom]
        :param dist_adj: [bsz, n_atom, n_atom]
        :return: attention bias with mask [bsz*n_head, n_atom, n_atom]
        """
        # bsz, n_hop, n_type, n_atom, _ = bond_adj.size()
        # bond_embed = self.edge_encoders[0](bond_adj[:, :, 0].int()).sum(dim=1)
        # for i in range(1, self.n_graph_type):
        #     bond_embed += self.edge_encoders[i](bond_adj[:, :, i].int()).sum(dim=1)

        # [bsz, n_hop, n_type, n_atom, n_atom, n_head] -> # [bsz, n_atom, n_atom, n_head]

        # bond_embed = torch.cat([
        #     bond_embed[:, i].unsqueeze(1).expand(bsz, hop, n_atom, n_atom)
        #     for i, hop in enumerate(self.hop_distribution)
        # ], dim=1)  # [bsz, n_hop, n_type, n_atom, n_atom] -> # [bsz, n_head, n_atom, n_atom]
        bsz, n_atom, _ = bond_adj.size()
        #print('n_atom:',n_atom)
        comb_embed = torch.zeros(bsz, n_atom, n_atom, self.num_heads, device=bond_adj.device)
        if self.use_dist_adj and dist_adj is not None:
            comb_embed += self.spatial_encoder(dist_adj)
        if self.use_3d_info and dist3d_adj is not None:
            comb_embed += self.spatial3d_encoder(dist3d_adj)


        if self.max_single_hop > 0:
            for i in range(self.n_graph_type):
                j_hop_embed = bond_adj.long()
                # decode to multi sense embedding
                j_hop_embed = torch.where(j_hop_embed > 0, ((j_hop_embed - 1) >> i) % 2, 0).float()
                base_hop_embed = j_hop_embed
                comb_embed += self.edge_encoders[i](j_hop_embed.int())
                for j in range(1, self.max_single_hop):
                    # generate multi atom environment embedding
                    j_hop_embed = torch.bmm(j_hop_embed, base_hop_embed)
                    comb_embed += self.edge_encoders[i](j_hop_embed.int())

        comb_embed = comb_embed.permute(0, 3, 1, 2)
        mask = torch.where(bond_adj != 0, 0., -torch.inf)

        if self.need_graph_token:
            graph_token = self.graph_token.weight[0].clone()
            if contrast:
                graph_token += self.contrast_token.weight[0]
            graph_token = graph_token.view(1, self.num_heads, 1, 1).repeat(bsz, 1, 1, 1)

            comb_embed = torch.cat([graph_token.expand(bsz, self.num_heads, n_atom, 1),
                                    comb_embed], dim=-1)
            comb_embed = torch.cat([graph_token.expand(bsz, self.num_heads, 1, n_atom + 1),
                                    comb_embed], dim=-2)

            mask = torch.cat([torch.zeros_like(mask[:, :, 0]).unsqueeze(-1), mask], dim=-1)
            mask = torch.cat([torch.zeros_like(mask[:, 0, :]).unsqueeze(-2), mask], dim=-2)

            # [bsz, n_head, n_atom+1, n_atom+1]
            n_atom += 1
        mask = mask.unsqueeze(1).expand(bsz, self.num_heads, n_atom, n_atom)
        out=(comb_embed + mask).reshape(bsz * self.num_heads, n_atom, n_atom)
        return out

#4.graphormer的模型主题部分
class GraphormerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(GraphormerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm.cuda()

    def forward(self, x, attn_mask=None):
        output = x
        for mod in self.layers:
            output = mod(output, attn_mask)

        if self.norm is not None and self.num_layers > 0:
            #print(output.is_cuda)
            #print(self.norm(output).is_cuda)
            output = self.norm(output)
        return output

class GraphormerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, device='cuda',batch_first=False, norm_first=False) -> None:
        super(GraphormerEncoderLayer, self).__init__()
        self.device=device
        self.self_attn = MultiHeadAtomAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,device=device)

        self.linear1 = nn.Linear(d_model, dim_feedforward).cuda()
        self.dropout = nn.Dropout(dropout).cuda()
        self.linear2 = nn.Linear(dim_feedforward, d_model).cuda()

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps).cuda()
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps).cuda()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(GraphormerEncoderLayer, self).__setstate__(state)

    def forward(self, x, attn_mask=None):
        #print((self.norm1(x + self._sa_block(x, attn_mask))).is_cuda)
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class MultiHeadAtomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,device='cuda',add_zero_attn=False,
                 batch_first=False) -> None:
        super(MultiHeadAtomAttention, self).__init__()
        self.device=device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)).to(device))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim).to(device))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias).cuda()

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        q_d, bsz, _ = q.size()
        k_d = k.size(0)
        q = q.contiguous().view(q_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # if attn_mask is not None and attn_mask.size(0) != bsz * self.num_heads:
        #     attn_mask = attn_mask.reshape(bsz, 1, q_d, k_d) \
        #         .expand(bsz, self.num_heads, q_d, k_d).reshape(bsz * self.num_heads, q_d, k_d)

        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)

        dropout_p = self.dropout if self.training else 0.0
        attn_output, attn_output_weights = \
            _scaled_dot_product_atom_attention(q, k, v, dropout_p, attn_mask)

        attn_output = attn_output.transpose(0, 1).contiguous().view(q_d, bsz, self.embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, q_d, k_d)
            return attn_output, attn_output_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return attn_output, None
def _scaled_dot_product_atom_attention(q, k, v=None, dropout_p=0.0, attn_mask=None):
    """
    :param attn_mask:
    :param q: [bsz, q, d]
    :param k: [bsz, k, d]
    :param v: [bsz, k, d]
    :param dropout_p: p in [0, 1]
    :return:([bsz, q, d], [bsz, q, k]) or (None, [bsz, q, k]) if v is None
    """
    B, Q, D = q.size()
    # print("q.size:", q.size(), k.size())
    q = q / math.sqrt(D)
    attn = torch.bmm(q, k.transpose(-2, -1))
    #这里更改了，感觉要是要对齐要大改
    #if attn_mask is not None:
    #    attn += attn_mask
        # attn = torch.nan_to_num(attn)
    if v is not None:
        attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    output = torch.bmm(attn, v) if v is not None else None
    # print(f"q:{q}\n"
    #       f"k:{k}\n"
    #       f"v:{v}\n"
    #       f"attn_mask:{attn_mask}"
    #       f"attn:{attn}")
    # raise ValueError
    return output, attn
def _in_projection_packed(q, k, v=None, w=None, b=None):
    """
    :param q: [q, bsz, d]
    :param k: [k, bsz, d]
    :param v: [v, bsz, d]
    :param w: [d*3, d]
    :param b: [d*3]
    :return: projected [q, k, v] or [q, k] if v is None
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    elif v is not None:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    elif q is k:
        return F.linear(q, w, b).chunk(2, dim=-1)
    else:
        w_q, w_k = w.split([E, E])
        if b is None:
            b_q = b_k = None
        else:
            b_q, b_k = b.split([E, E])
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k)

class MultiHeadAtomAdj(nn.Module):
        def __init__(self, embed_dim, num_heads, bias=True, device='cuda',add_zero_attn=False,
                     batch_first=False) -> None:
            super(MultiHeadAtomAdj, self).__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.batch_first = batch_first
            self.head_dim = embed_dim // num_heads
            self.device=device
            assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
            self.in_proj_weight = nn.Parameter(torch.empty((2 * embed_dim, embed_dim)).to(device))
            if bias:
                self.in_proj_bias = nn.Parameter(torch.empty(2 * embed_dim).to(device))
            else:
                self.register_parameter('in_proj_bias', None)
            self.add_zero_attn = add_zero_attn
            self._reset_parameters()
        def _reset_parameters(self):
            xavier_uniform_(self.in_proj_weight)
            if self.in_proj_bias is not None:
                constant_(self.in_proj_bias, 0.)
        def forward(self, query, key, attn_mask=None):
            if self.batch_first:
                query, key = [x.transpose(1, 0) for x in (query, key)]
            q, k = _in_projection_packed(query, key, None, self.in_proj_weight, self.in_proj_bias)
            q_d, bsz, _ = q.size()
            k_d = k.size(0)
            q = q.contiguous().view(q_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            k = k.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)

            # if attn_mask is not None and attn_mask.size(0) != bsz * self.num_heads:
            #     attn_mask = attn_mask.reshape(bsz, 1, q_d, k_d) \
            #         .expand(bsz, self.num_heads, q_d, k_d).reshape(bsz * self.num_heads, q_d, k_d)

            if self.add_zero_attn:
                zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
                k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)

            _, adj = _scaled_dot_product_atom_attention(q, k, None, 0.0, attn_mask)

            return adj.view(bsz, self.num_heads, q_d, k_d).permute(0, 2, 3, 1).contiguous()
#5.整个drug处理部分形成一个统一的encoder,直接掉用他放在predict里！

class DrugEncoder(nn.Module):
    def __init__(self,d_model=512,nhead=32,num_layer=8,dim_feedforward=512,dropout=0.1,atom_dim=92,total_degree=10,hybrid=6,hydrogen=9,aromatic=2,ring=10,n_layers=1,
                 max_paths=50, n_graph_type=6, max_single_hop=4,
                 activation=F.gelu,use_3d_info=False,use_dist_adj=True,need_graph_token=True,batch_first=True,norm_first=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_layer
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.use_3d_info = use_3d_info
        self.use_dist_adj = use_dist_adj
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.n_graph_type = n_graph_type
        self.max_single_hop = max_single_hop
        self.max_paths=max_paths
        encoder_layer = GraphormerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                batch_first=batch_first, norm_first=norm_first)
        self.emb = GraphormerEmbeddingLayer(d_model, nhead, max_paths, n_graph_type, max_single_hop, atom_dim, total_degree,
                 hybrid, hydrogen, aromatic, ring, n_layers, need_graph_token=True, use_3d_info=use_3d_info,
                                      dropout=dropout,use_dist_adj=use_dist_adj)
        self.encoder = GraphormerEncoder(encoder_layer, num_layer, nn.LayerNorm(d_model))
    def forward(self,atom_fea,bond_adj,dist_adj):
        """{
            atom_fea:[bsz, n_type, n_atom]
            bond_adj: [bsz, n_atom, n_atom]
            dist_adj:
            center_cnt:
            rxn_type:
            dist3d_adj:
            lg_dic:
        }
        :return:
        """
        bsz, n_atom, _ = bond_adj.size()
        atom_fea, masked_adj = self.emb(atom_fea, bond_adj, dist_adj)
        fea = self.encoder(atom_fea, masked_adj)
        #fea = self.encoder(shared_atom_fea, masked_adj)
        #print('drug embedding:',fea.shape)
        return fea

#不知道要干什么，反正需要进行参数的初始化就对了
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#以下是decoder部分，获取intection vector
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask=None):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        d_k=Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / \
            np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        # Fills elements of self tensor with value where mask is True.
        #scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        # context: [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        #print('attention:',attn.shape)
        return context, attn

class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    输入：seq_len x d_model
    输出：seq_len x d_model
    """

    def __init__(self,d_model,n_heads,dropout,device):
        super(MultiHeadAttention, self).__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.device=device
        self.W_Q = nn.Linear(d_model, d_model * n_heads,
                             bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_model * n_heads, bias=False)
        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        self.fc = nn.Linear(n_heads * d_model, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1,
                                   self.n_heads, self.d_model).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1,
                                   self.n_heads, self.d_model).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1,
                                   self.n_heads, self.d_model).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        #attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        attention_map = torch.mean(attn, [0, 1])
        attentions=torch.mean(attn,[1])
        '''
        at=pd.DataFrame()
        for i in range(len(attn)):
            at=pd.DataFrame(attentions[i].cpu().detach().numpy())
            at.to_csv('/mnt/sdb/home/hjy/exp1/file/attention_analysis'+str(i)+'.csv', encoding='utf_8_sig')
            print('save end')
        '''
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.n_heads * self.d_model)

        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual), attention_map

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        # query = key = value [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch size, sent len, hid dim]
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        print("q:",Q.shape)
        print("k:",K.shape)
        #print(Q.shape)
        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #print('mask:',mask)
        # energy = [batch size, n heads, sent len_Q, sent len_K]
        #if mask is not None:
        #    energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        attention_map=torch.mean(attention,[0,1])
        print('attention:',attention.shape)
        #attention_map=attention[0,0,:,:]
        # attention = [batch size, n heads, sent len_Q, sent len_K]
        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch size, src sent len_Q, hid dim]
        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim]
        return x,attention_map

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]
        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]
        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch size, pf dim, sent len]
        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]
        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]
        trg,_=self.sa(trg, trg, trg, trg_mask)

        trg = self.ln(trg + self.do(trg))

        trg,a=self.ea(trg, src, src, src_mask)

        trg = self.ln(trg + self.do(trg))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg,a


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        #self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)
        self.gn = nn.GroupNorm(8, 256)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)
        attention=[]
        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg,attention_tmp= layer(trg, src,trg_mask,src_mask)
            attention.append(attention_tmp)

        #attention = [sent len_q,sent len_k]
        attention=attention[-1]
        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg, dim=2)
        # norm = [batch size,compound len]
        norm = F.softmax(norm, dim=1)
        # norm = [batch size,compound len]
        # trg = torch.squeeze(trg,dim=0)
        # norm = torch.squeeze(norm,dim=0)
        sum = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j, ]
                v = v * norm[i, j]
                sum[i, ] += v
        # sum = [batch size,hid_dim]
        label = F.relu(self.fc_1(sum))
        label = self.fc_2(label)
        return label,sum,attention

