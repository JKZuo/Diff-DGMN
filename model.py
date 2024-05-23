import gol
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence

from layers import SeqGraphEncoder, DisGraphRep, SDE_Diffusion, PFFN

'''Identify the actual length and padding of the user check-in trajectory sequence'''
def Seq_MASK(lengths, max_len=None): 
    lengths_shape = lengths.shape  
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device).type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len).lt(lengths.unsqueeze(1))).reshape(lengths_shape)

 
class DiffDGMN(nn.Module):
    def __init__(self, n_user, n_poi, G_D: Data):
        super(DiffDGMN, self).__init__()
        self.n_user, self.n_poi = n_user, n_poi
        self.hid_dim = gol.conf['hidden']
        self.step_num = 1000

        # Initialize all parameters
        self.poi_emb = nn.Parameter(torch.empty(n_poi, self.hid_dim))
        self.delta_dis_embs = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        self.delta_time_embs = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        nn.init.xavier_normal_(self.poi_emb)
        nn.init.xavier_normal_(self.delta_dis_embs)
        nn.init.xavier_normal_(self.delta_time_embs)

        self.seq_Rep = SeqGraphEncoder(self.hid_dim)
        self.dis_Rep = DisGraphRep(n_poi, self.hid_dim, G_D)
        self.SDEdiff = SDE_Diffusion(self.hid_dim, beta_min=gol.conf['beta_min'], beta_max=gol.conf['beta_max'], dt=gol.conf['dt'])

        self.CEloss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=gol.conf['dp']) 

        # Initialize Module (A) SeqGraphRep 
        self.seq_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn = nn.MultiheadAttention(embed_dim = self.hid_dim, num_heads=gol.conf['num_heads'], batch_first=True, dropout=0.2)
        self.seq_PFFN = PFFN(self.hid_dim, 0.2)

        # Initialize Module (B) DisGraphRep and Module (C) LocaGenerator
        self.geo_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn = nn.MultiheadAttention(embed_dim = self.hid_dim, num_heads=gol.conf['num_heads'], batch_first=True, dropout=0.2)
        self.geo_PFFN = PFFN(self.hid_dim, 0.2)

    '''
    SeqGraphRep: (A) Direction-aware Sequence Graph Multi-scale Representation Module in DiffDGMN
        Input: User-oriented POI Transition Graph G_u
        Return: Sequence Encoding of user S_u
    '''
    def SeqGraphRep(self, POI_embs, G_u):
        seq_embs = self.seq_Rep.encode((POI_embs, self.delta_dis_embs, self.delta_time_embs), G_u)

        if gol.conf['dropout']:
            seq_embs = self.dropout(seq_embs)

        seq_lengths = torch.bincount(G_u.batch)
        seq_embs = torch.split(seq_embs, seq_lengths.cpu().numpy().tolist())

        seq_embs_pad = pad_sequence(seq_embs, batch_first=True, padding_value=0)  
        pad_mask = Seq_MASK(seq_lengths)  
        
        # Q, K, V = [seq_length, batch_size, embed_dim] 
        Q = self.seq_layernorm(seq_embs_pad)                                                                    
        K = seq_embs_pad
        V = seq_embs_pad
        output, att_weights = self.seq_attn(Q, K, V, key_padding_mask=~pad_mask)

        output = output + Q
        output = self.seq_attn_layernorm(output)

        # PFNN
        output = self.seq_PFFN(output)           
        output = [seq[:seq_len] for seq, seq_len in zip(output, seq_lengths)]

        S_u = torch.stack([seq.mean(dim=0) for seq in output], dim=0)
        return S_u

    '''
    LocaGenerator: (C) Attention-based Location Archetype Generation Module in DiffDGMN
        Input: Sequence Encoding of user S_u, Node Geographical Representation R_V
        Return: Location Archetype Vector hat_L_u
    '''
    def LocaGenerator(self, POI_embs, seqs, S_u):
        # Node Geographical Representation R_V
        R_V = self.dis_Rep.encode(POI_embs) 
        if gol.conf['dropout']:
            R_V = self.dropout(R_V)
        
        seq_lengths = torch.LongTensor([seq.size(0) for seq in seqs]).to(gol.device)
        R_V_seq = [R_V[seq] for seq in seqs]


        R_V_pad = pad_sequence(R_V_seq, batch_first=True, padding_value=0)
        pad_mask = Seq_MASK(seq_lengths)
        Q = self.geo_layernorm(S_u.detach().unsqueeze(1))
        K = R_V_pad
        V = R_V_pad

        output, att_weights = self.geo_attn(Q, K, V, key_padding_mask=~pad_mask)

        # output = output + Q
        output = output.squeeze(1)
        output = self.geo_attn_layernorm(output)

        hat_L_u = self.geo_PFFN(output)   # PFNN
        return hat_L_u, R_V

    '''
    DiffGenerator: (D) Diffusion-based User Preference Sampling Module in DiffDGMN
        Input: Location Archetype Vector hat_L_u, Sequence Encoding S_u as a context-aware condition embedding
        Return: A Pure (noise-free) Location Archetype Vector L_u
    '''
    def DiffGenerator(self, hat_L_u, S_u, target=None):
        local_embs = hat_L_u
        condition_embs = S_u.detach()

        # Reverse-time VP-SDE Generation Process
        L_u = self.SDEdiff.ReverseSDE_gener(local_embs, condition_embs, gol.conf['T'])

        loss_div = None
        if target is not None: # training phase
            t_sampled = np.random.randint(1, self.step_num) / self.step_num

            # get marginal probability
            mean, std = self.SDEdiff.marginal_prob(target, t_sampled)
            z = torch.randn_like(target)
            perturbed_data = mean + std.unsqueeze(-1) * z

            # train a time-dependent score-based neural network to estimate marginal probability
            score = - self.SDEdiff.Est_score(perturbed_data, condition_embs)

            # Fisher divergence loss_div
            loss_div = torch.square(score + z).mean()

        return L_u, loss_div



    '''Get cross-entropy recommendation loss and Fisher divergence loss'''
    def getTrainLoss(self, batch):
        usr, pos_lbl, exclude_mask, seqs, G_u, cur_time = batch

        R_v = self.poi_emb
        POI_embs = self.poi_emb
        if gol.conf['dropout']:
            POI_embs = self.dropout(POI_embs)

        S_u = self.SeqGraphRep(POI_embs, G_u)
        hat_L_u, R_V = self.LocaGenerator(POI_embs, seqs, S_u)
        L_u, loss_div = self.DiffGenerator(hat_L_u, S_u, target=R_V[pos_lbl])

        Y_predscore = 2 * torch.matmul(S_u, R_v.t()) + torch.matmul(L_u, R_V.t())

        loss_rec = self.CEloss(Y_predscore, pos_lbl)
        return loss_rec, loss_div  


    def forward(self, seqs, G_u):
        R_v = self.poi_emb
        POI_embs = self.poi_emb

        S_u = self.SeqGraphRep(POI_embs, G_u)
        hat_L_u, R_V = self.LocaGenerator(POI_embs, seqs, S_u)
        L_u, _ = self.DiffGenerator(hat_L_u, S_u)

        '''
        Y_predscore = [batch_size, #POI]
        S_u         = [batch_size, hid_dim]
        R_v.T       = [hid_dim, #POI] 
        L_u         = [batch_size, hid_dim]
        R_V.T       = [hid_dim, #POI]
        '''

        Y_predscore = 2 * torch.matmul(S_u, R_v.t()) + torch.matmul(L_u, R_V.t()) 
        return Y_predscore
    
