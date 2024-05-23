import gol
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, softmax
from torchsde import sdeint

'''
PFFN: Point-wise Feed Forward Network
'''
class PFFN(nn.Module):
    def __init__(self, hid_size, dropout_rate):
        super(PFFN, self).__init__()
        self.conv1 = nn.Conv1d(hid_size, hid_size, kernel_size=1) 
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hid_size, hid_size, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs



'''
BiSeqGCN: Bi-directional Sequence Graph Convolution, is as a part of (A) Direction-aware Sequence Graph Multi-scale Representation Module (SeqGraphRep)
    Input: User-oriented POI Transition Graph G_u
    Return: Node representation of the user-oriented POI transition graph H_u
'''
class BiSeqGCN(MessagePassing):
    def __init__(self, hid_dim, flow="source_to_target"):
        super(BiSeqGCN, self).__init__(aggr='add', flow=flow)
        self.hid_dim = hid_dim
        self.alpha_src = nn.Linear(hid_dim, 1, bias=False)
        self.alpha_dst = nn.Linear(hid_dim, 1, bias=False)
        
        # attention_weight
        self.attention_weight = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        nn.init.xavier_uniform_(self.attention_weight.data)
        
        nn.init.xavier_uniform_(self.alpha_src.weight)
        nn.init.xavier_uniform_(self.alpha_dst.weight)
        self.act = nn.LeakyReLU()

    def forward(self, embs, G_u):
        POI_embs, delta_dis_embs, delta_time_embs = embs
        sess_idx   = G_u.x.squeeze()
        edge_index = G_u.edge_index
        edge_time  = G_u.edge_time
        edge_dist  = G_u.edge_dist
        
        x = POI_embs[sess_idx]
        edge_l = delta_dis_embs[edge_dist]
        edge_t = delta_time_embs[edge_time]
        all_edges = torch.cat((edge_index, edge_index[[1, 0]]), dim=-1)
        
        H_u = self.propagate(all_edges, x=x, edge_l=edge_l, edge_t=edge_t, edge_size=edge_index.size(1))
        return H_u

    def message(self, x_j, x_i, edge_index_j, edge_index_i, edge_l, edge_t, edge_size):
        attention_coefficients = torch.matmul(x_i[edge_size:] + edge_l + edge_t, self.attention_weight.t())
        
        src_attention = self.alpha_src(attention_coefficients[:edge_size]).squeeze(-1)
        dst_attention = self.alpha_dst(attention_coefficients[:edge_size]).squeeze(-1)
        
        # softmax on tot_attention
        tot_attention = torch.cat((src_attention, dst_attention), dim=0)
        attn_weight = softmax(tot_attention, edge_index_i)

        # attn_weight on neighbor node features
        updated_rep = x_j * attn_weight.unsqueeze(-1)
        return updated_rep

'''
SeqGraphEncoder: Encode BiSeqGCN, is as a part of (A) Direction-aware Sequence Graph Multi-scale Representation Module (SeqGraphRep)
'''
class SeqGraphEncoder(nn.Module):
    def __init__(self, hid_dim):
        super(SeqGraphEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.encoder = BiSeqGCN(hid_dim)

    def encode(self, embs, G_u):
        return self.encoder(embs, G_u)



'''
DisDyGCN: Distance-based Dynamic Graph Convolution, is as a part of (B) Global-based Distance Graph Geographical Representation Module (DisGraphRep)
    Input: Global-based POI Distance Graph G_D
    Return: Updated Node Information h
'''
class DisDyGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, dist_embed_dim=64):
        super(DisDyGCN, self).__init__(aggr='add')
        self._cached_edge = None
        self.linear = nn.Linear(in_channels, out_channels)
        nn.init.xavier_uniform_(self.linear.weight)
        
        # dynamic mechanism on diatance
        self.dist_transform = nn.Sequential(
            nn.Linear(1, dist_embed_dim),  
            nn.ReLU(),
            nn.Linear(dist_embed_dim, out_channels)  
            )
        
        nn.init.xavier_uniform_(self.dist_transform[0].weight)
        nn.init.xavier_uniform_(self.dist_transform[2].weight)

    def forward(self, x, G_D: Data):
        if self._cached_edge is None:
            self._cached_edge = gcn_norm(G_D.edge_index, add_self_loops=False)
        edge_index, norm_weight = self._cached_edge
        x = self.linear(x)
        h = self.propagate(edge_index, x=x, norm=norm_weight, dist_vec=G_D.edge_attr)
        return h
    
    def message(self, x_j, norm, dist_vec):
        dist_weight = self.dist_transform(dist_vec.unsqueeze(-1))
        message_trans = norm.unsqueeze(-1) * x_j * dist_weight
        return message_trans

'''
DisGraphRep: (B) Global-based Distance Graph Geographical Representation Module in DiffDGMN
    Input: Global-based POI Distance Graph G_D
    Return: Node Geographical Representation R_V
'''
class DisGraphRep(nn.Module):
    def __init__(self, n_poi, hid_dim, G_D: Data):
        super(DisGraphRep, self).__init__()
        self.n_poi, self.hid_dim = n_poi, hid_dim
        self.GCN_layer = gol.conf['num_layer']

        # aggregating own features: 
        edge_index, _ = add_self_loops(G_D.edge_index)  
        dist_vec = torch.cat([G_D.edge_attr, torch.zeros((n_poi,)).to(gol.device)])
        # a_{i,j}^D: 
        dis_edgeweight = torch.exp(-(dist_vec ** 2)) 
        self.G_D = Data(edge_index=edge_index, edge_attr=dis_edgeweight)

        self.act = nn.LeakyReLU()
        self.DisDyGCN = nn.ModuleList()
        for _ in range(self.GCN_layer):
            self.DisDyGCN.append(DisDyGCN(self.hid_dim, self.hid_dim))

    def encode(self, poi_embs):
        layer_embs = poi_embs
        geo_embs = [layer_embs]
        
        for conv in self.DisDyGCN:
            layer_embs = conv(layer_embs, self.G_D) 
            layer_embs = self.act(layer_embs)
            geo_embs.append(layer_embs)

        R_V = torch.stack(geo_embs, dim=1).mean(1)
        return R_V  



'''
SDEsolver: Stochastic Differential Equation Solver 
'''
class SDEsolver(nn.Module):
    sde_type = 'stratonovich'   # available: {'ito', 'stratonovich'}
    noise_type = 'scalar'       # available: {'general', 'additive', 'diagonal', 'scalar'}

    def __init__(self, f, g):
        super(SDEsolver).__init__()
        self.f, self.g = f, g

    def f(self, t, y): 
        return self.f(t, y)
    
    def g(self, t, y): 
        return self.g(t, y)

'''
SDE_Diffusion: Stochastic-Differential-Equation-based Diffusion unit, as a part of (D) Diffusion-based User Preference Sampling (DiffGenerator)
    Input: Location Archetype Vector hat_L_u, Sequence Encoding S_u as a context-aware condition embedding
    Return: A Pure (noise-free) Location Archetype Vector L_u, estimate marginal probability
'''
class SDE_Diffusion(nn.Module):
    def __init__(self, hid_dim, beta_min, beta_max, dt):
        super(SDE_Diffusion, self).__init__()
        self.hid_dim = hid_dim
        self.beta_min, self.beta_max = beta_min, beta_max
        self.dt = dt
        
        # score-based neural network stacked multiple fully connected layers
        self.score_FC = nn.Sequential(
                    nn.Linear(2 * hid_dim, 2 * hid_dim),
                    nn.BatchNorm1d(2 * hid_dim),
                    nn.ReLU(inplace=True),
                    # nn.Dropout(p=0.2), 
                    nn.Linear(2 * hid_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.ReLU(inplace=True),
                    # nn.Dropout(p=0.2), 
                    nn.Linear(hid_dim, hid_dim)
                    )
        
        for w in self.score_FC:
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

    # a time-dependent score-based neural network to estimate marginal probability
    def Est_score(self, x, condition):
        # this score is used in SDE solving to guide the evolution of stochastic processes
        return self.score_FC(torch.cat((x, condition), dim=-1))

    #  Define the drift term f and diffusion term g of Forward SDE
    def ForwardSDE_diff(self, x, t):
        def f(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            return -0.5 * beta_t * y

        def g(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs = y.size(0)
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            return (beta_t ** 0.5) * noise

        ts = torch.Tensor([0, t]).to(gol.device)
        output = sdeint(SDEsolver(f, g), y0=x, ts=ts, dt=self.dt)[-1]
        return output

    def ReverseSDE_gener(self, x, condition, T):
        def get_beta_t(_t):
            beta_t_1 = self.beta_min + _t * (self.beta_max - self.beta_min)
            beta_t_2 = self.beta_min + (self.beta_max - self.beta_min) * torch.sin(torch.pi/2 * _t )**2
            beta_t_3 = self.beta_min * torch.exp(torch.log(torch.tensor(self.beta_max / self.beta_min)) * _t)
            beta_t_4 = self.beta_min + _t * (self.beta_max - self.beta_min)**2
            beta_t_5 = 0.1 * torch.exp(6 * _t)
            return beta_t_1
        
        # drift term f(): {_t: current time point, y: current state, returns the value of the drift term}
        def f(_t, y): 
            beta_t = get_beta_t(_t)
            score = self.score_FC(torch.cat((x, condition), dim=-1))
            ## score = self.score_FC(y)
            drift = -0.5 * beta_t * y - beta_t * score
            return drift
        
        # diffusion term g(): {_t: current time point, y: current state, returns the value of the diffusion term}
        def g(_t, y): 
            beta_t = get_beta_t(_t)
            bs = y.size(0)
            # noise Tensors [bs, self.hid_dim, 1] = [1024, 64, 1]  with all elements of 1
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device) 
            diffusion = (beta_t ** 0.5)  * noise
            return diffusion
        
        def g_diagonal_noise(_t, y): 
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim = y.size(0), y.size(1)
            noise = torch.randn((bs, dim), device=y.device)
            diagonal_beta = torch.diag(beta_t * torch.ones(dim, device=y.device))  
            diffusion = (diagonal_beta ** 0.5).mm(noise.t()).t()
            return diffusion + y
        
        def g_vector_noise(_t, y): 
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim, brownian_size = y.size(0), y.size(1), y.size(1)
            noise = torch.randn((bs, dim, brownian_size), device=y.device)
            diffusion = (beta_t ** 0.5) * noise
            return diffusion
        
        def g_full_cov_noise_3d(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim = y.size(0), y.size(1)
            noise = torch.randn((bs, dim, dim), device=y.device) 
            covariance_matrix = torch.eye(dim, device=y.device)
            covariance_matrix = covariance_matrix * beta_t
            cholesky_matrix = torch.linalg.cholesky(covariance_matrix)
            diffusion = torch.einsum('bij,jk->bik', noise, cholesky_matrix)
            return diffusion

        ts = torch.Tensor([0, T]).to(gol.device) 

        # output is a pure (noise-free) location archetype vector L_u
        output = sdeint(SDEsolver(f, g), y0=x, ts=ts, dt=self.dt)[-1]

        return output

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        log_mean_coeff = torch.Tensor([log_mean_coeff]).to(x.device)
        mean = torch.exp(log_mean_coeff.unsqueeze(-1)) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std
