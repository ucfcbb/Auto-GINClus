import os
import sys
import ast
import torch
import torch.nn as nn
import time
import copy

from torch_geometric.data import InMemoryDataset, HeteroData
from torch_geometric.loader import DataLoader
from torch.nn import Sequential, Linear, ReLU, Softplus
from torch_geometric.nn import global_mean_pool

### For GINEConv model
from torch_geometric.nn import GINEConv, HeteroConv

### For modified GINstyle model
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
##from torch_geometric.nn import HeteroConv

import random
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling

from GVP_libr import _norm_no_nan, LayerNorm, Dropout, _VDropout, tuple_sum, tuple_cat, tuple_index, _split, _merge, GVP, GVPConv, GVPConvLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



########### Read input data to generate graph representation ###########
def read_graph(in_file):
            
    PDB_location, FASTA_location, graph_label = "", "", ""
    num_nodes = 0
    num_edges = 0
    node_features, distance_edge_src, distance_edge_des, distance_edge_features, bond_edge_src, bond_edge_des, bond_edge_features, orientation_edge_src, orientation_edge_des, orientation_edge_features, node_coordinates,\
    scalar_edge_features, edge_orientations = [], [], [], [], [], [], [], [], [], [], [], [], []
    
    while(True):
        line = in_file.readline()
        if line == "":
            break

        line = line.strip("\n")

        ### Read motif location
        if line[0:15] == "#Motif location":
            PDB_location = in_file.readline().strip("\n").split("\t")[0]
            FASTA_location = in_file.readline().strip("\n")

        ### Read motif sequence
        if line[0:15] == "#Motif sequence":
            SEQ = in_file.readline().strip("\n")

        ### Read number of nodes
        if line[0:7] == "#Number":
            num_nodes = int(in_file.readline().strip("\n"))

        ### Read node features
        if line[0:14] == "#Node_Features":
            for node in range(num_nodes):
                cur_node = in_file.readline().strip("\n").strip("\t")
                cur_node = cur_node.split("\t")
                cur_node = [int(element) for element in cur_node]
                node_features.append(cur_node)
            residue_embeddings = torch.tensor(node_features, dtype=torch.int32).to(device)

        ### Read node coordinates
        if line[0:17] == "#Node_Coordinates":
            for node in range(num_nodes):
                cur_node = in_file.readline().strip("\n").strip("\t")
                cur_node = cur_node.split("\t")
                cur_node = [float(element) for element in cur_node]
                node_coordinates.append([cur_node])
            node_coordinates = torch.tensor(node_coordinates, dtype=torch.float32).to(device)

        ### Read distance adjacency matrix
        if line[0:10] == "#Adjacency":
            for node_i in range(num_nodes):
                cur_line = in_file.readline().strip("\n").strip("\t")
                cur_line = cur_line.split("\t")
                for node_j in range(num_nodes):
                    distance_edge_src.append(node_i)
                    distance_edge_des.append(node_j)
                    RMSD = float(cur_line[node_j])
                    distance_edge_features.append([RMSD])
            scalar_edge_features = torch.tensor(distance_edge_features, dtype=torch.float32).to(device)

        ### Read bond edges
        if line[0:18] == "#Edge_Orientations":
            for node_i in range(num_nodes):
                cur_line = in_file.readline().strip("\n").strip("\t")
                cur_line = cur_line.split("\t")
                for node_j in range(num_nodes):
                    coor = cur_line[node_j].split(",")
                    x, y, z = float(coor[0]), float(coor[1]), float(coor[2])
                    edge_orientations.append([[x, y, z]])
            edge_orientations = torch.tensor(edge_orientations, dtype=torch.float32).to(device)

        ### Read bond edges
        if line[0:11] == "#Bond_Edges":
            cur_line = in_file.readline().strip("\n").strip("\t")
            bond_edges = ast.literal_eval(cur_line)
            for edge in bond_edges:
                bond_edge_src.append(edge[0])
                bond_edge_des.append(edge[1])

        ### Read bond edge features
        if line[0:19] == "#Bond_Edge_Features":
            for edge in range(len(bond_edges)):
                cur_line = in_file.readline().strip("\n").strip("\t")
                cur_line = cur_line.split("\t")
                cur_line = [int(element) for element in cur_line]
                bond_edge_features.append(cur_line)
            
        ### Read orientation edges
        if line[0:18] == "#Orientation_Edges":
            cur_line = in_file.readline().strip("\n").strip("\t")
            orn_edges = ast.literal_eval(cur_line)
            for edge in orn_edges:
                orientation_edge_src.append(edge[0])
                orientation_edge_des.append(edge[1])

        ### Read orientation edge features
        if line[0:26] == "#Orientation_Edge_Features":
            for edge in range(len(orn_edges)):
                cur_line = in_file.readline().strip("\n").strip("\t")
                cur_line = cur_line.split("\t")
                cur_line = [int(element) for element in cur_line]
                orientation_edge_features.append(cur_line)

        ### Read label
        if line[0:12] == "#Graph_label":
            graph_label = in_file.readline().strip("\n").strip("\t")
            
    in_file.close()


    ### Randomly mask residue embeddings for GVP
    residue_embeddings_masked, node_mask = mask_residue_embeddings(residue_embeddings, 0.4)


    ### Edge indexes
    ### Node features (scalar, vector)
    # h_V = (residue_embeddings, node_coordinates)
    h_V = (residue_embeddings_masked, node_coordinates)
    ### Edge features (scalar, vector)
    h_E = (scalar_edge_features, edge_orientations)
    ### Sequence
    letter_to_num = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    seq = torch.as_tensor([letter_to_num[a] for a in SEQ], device=device, dtype=torch.long)
    valid_node_mask = torch.isfinite(node_coordinates.sum(dim=(1,2))) ### Removes nodes with NaN or InF coordinate
    mask = node_mask & valid_node_mask # combining both masks

    # mask = torch.isfinite(node_coordinates.sum(dim=(1,2))) ### Removes nodes with NaN or InF coordinate
    return num_nodes, node_features, distance_edge_src, distance_edge_des, distance_edge_features, bond_edge_src, bond_edge_des, bond_edge_features, orientation_edge_src, orientation_edge_des, orientation_edge_features, graph_label, PDB_location, h_V, h_E, seq, mask
########### Read input data to generate graph representation ###########


########### Building class MyDataset ###########
class MyDataset(InMemoryDataset):
    def __init__(self, root, num_graphs, Graph_dic, transform=None, pre_transform=None):
        
        self.num_graphs = num_graphs
        self.Graph_dic = Graph_dic

        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], map_location=torch.device("cpu"), weights_only=False)

    @property
    def raw_file_names(self):
        return ['dummy.txt']  # Not used

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []

        for graph_no in range(self.num_graphs):
            num_nodes, node_features, distance_edge_src, distance_edge_des, distance_edge_features, bond_edge_src, bond_edge_des, bond_edge_features, orientation_edge_src, orientation_edge_des,\
            orientation_edge_features, graph_label, PDB_location, h_V, h_E, seq, mask = self.Graph_dic[graph_no][0], self.Graph_dic[graph_no][1], self.Graph_dic[graph_no][2], self.Graph_dic[graph_no][3], self.Graph_dic[graph_no][4],\
            self.Graph_dic[graph_no][5], self.Graph_dic[graph_no][6], self.Graph_dic[graph_no][7], self.Graph_dic[graph_no][8], self.Graph_dic[graph_no][9], self.Graph_dic[graph_no][10], self.Graph_dic[graph_no][11], self.Graph_dic[graph_no][12],\
            self.Graph_dic[graph_no][13], self.Graph_dic[graph_no][14], self.Graph_dic[graph_no][15], self.Graph_dic[graph_no][16]
            data = HeteroData()

            # Unique graph id
            data.graph_id = torch.tensor([graph_no]).to(device)

            # Create node features
            data['nucleotide'].x = torch.tensor(node_features, dtype=torch.float).to(device)       # 13 nucleotides, 4 features

            # Edges: bond edges
            data['nucleotide', 'bonds_with', 'nucleotide'].edge_index = torch.tensor([
                bond_edge_src,  # nucleotide indices
                bond_edge_des   # nucleotide indices
            ], dtype=torch.long).to(device)

            # Edge attributes: bond edge attributes
            data['nucleotide', 'bonds_with', 'nucleotide'].edge_attr = torch.tensor(
                bond_edge_features, dtype=torch.float).to(device)

            # Edges: orientation edges
            data['nucleotide', 'orientation_to', 'nucleotide'].edge_index = torch.tensor([
                orientation_edge_src,  # nucleotide indices
                orientation_edge_des   # nucleotide indices
            ], dtype=torch.long).to(device)

            # Edge attributes: orientation edge attributes
            data['nucleotide', 'orientation_to', 'nucleotide'].edge_attr = torch.tensor(
                orientation_edge_features, dtype=torch.float).to(device)

            # Edges: distance edges (self-type)
            data['nucleotide', 'distance_from', 'nucleotide'].edge_index = torch.tensor([
                distance_edge_src,
                distance_edge_des
            ], dtype=torch.long).to(device)

            # Edge attributes: distance edge attributes
            data['nucleotide', 'distance_from', 'nucleotide'].edge_attr = torch.tensor(
                distance_edge_features, dtype=torch.float).to(device)

            ### Adding features for GVP
            gvp_x = h_V
            
            data['nucleotide'].gvp_x = h_V
            data['nucleotide'].gvp_edge_attr = h_E
            data['nucleotide'].seq = seq
            data['nucleotide'].mask = mask
            
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
########### Building class MyDataset ###########        



########### Autoencoder GINEConv implement for multiple hetergenous graphs ###########
###### Encoder ######
class HeteroGINEEncoder(torch.nn.Module):
    def __init__(self, metadata, input_node_dim, hidden_dim, edge_dims):
        super().__init__()

        self.node_lin_dict = torch.nn.ModuleDict({
            node_type: Linear(input_node_dim, hidden_dim)
            for node_type in metadata[0]
        })

        self.convs = HeteroConv({
            edge_type: GINEConv(
                nn=Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim)
                ),
                edge_dim=edge_dims[edge_type]
            )
            for edge_type in metadata[1]
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = {
            k: self.node_lin_dict[k](v)
            for k, v in x_dict.items()
        }
        return self.convs(x_dict, edge_index_dict, edge_attr_dict)



##### GINEEncoder with 3 layers
# class HeteroGINEEncoder(nn.Module):
#     def __init__(self, metadata, input_node_dim, hidden_dim, edge_dims, num_layers):
#         super().__init__()

#         # Initial linear transform for each node type
#         self.node_lin_dict = nn.ModuleDict({
#             node_type: nn.Linear(input_node_dim, hidden_dim)
#             for node_type in metadata[0]
#         })

#         # Multiple GINEConv layers
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             conv = HeteroConv({
#                 edge_type: GINEConv(
#                     nn=nn.Sequential(
#                         nn.Linear(hidden_dim, hidden_dim),
#                         nn.ReLU(),
#                         nn.Linear(hidden_dim, hidden_dim)
#                     ),
#                     edge_dim=edge_dims[edge_type]
#                 )
#                 for edge_type in metadata[1]
#             }, aggr='sum')
#             self.layers.append(conv)

#         self.activation = nn.ReLU()

#     def forward(self, x_dict, edge_index_dict, edge_attr_dict):
#         # Initial projection of node features
#         x_dict = {
#             k: self.node_lin_dict[k](v)
#             for k, v in x_dict.items()
#         }

#         # Apply HeteroGINE layers with activation
#         for conv in self.layers:
#             x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
#             x_dict = {k: self.activation(v) for k, v in x_dict.items()}

#         return x_dict



###### Autoencoder module (For edge attribute reconstruction)) ######
class HeteroGraphAutoEncoder(torch.nn.Module):
    def __init__(self, metadata, input_node_dim, hidden_dim, edge_dims):
        super().__init__()
        self.encoder = HeteroGINEEncoder(metadata, input_node_dim, hidden_dim, edge_dims)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        z_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        
        return z_dict
########### Autoencoder GINEConv implement for multiple hetergenous graphs ###########    


########### GVP Model run ###########   
class GVPModel(torch.nn.Module):

    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.1):
    
        super(GVPModel, self).__init__()
        
        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )
        
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        self.W_s = nn.Embedding(4, 4)  ### Should it be (4, 4)?
        edge_h_dim = (edge_h_dim[0] + 4, edge_h_dim[1])
      
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, 
                             drop_rate=drop_rate, autoregressive=True) 
            for _ in range(num_layers))

        self.W_out = GVP(node_h_dim, (4, 0), activations=(None, None)) ### For 4 nucleotide type [A, C, U, T]

    def forward(self, h_V, edge_index, h_E, seq, train):
        '''
        Forward pass to be used at train-time, or evaluating likelihood.
        
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: int `torch.Tensor` of shape [num_nodes]
        '''
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        encoder_embeddings = h_V

        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x = encoder_embeddings)

        if train:
            logits = self.W_out(h_V)
            return logits
        
        else:
            sc, Ve = h_V  # s: [N, d_s], V: [N, d_v, 3]
           
            # Vector to scalar conversion using L2-norm
            V_norm = torch.norm(Ve, dim=-1)   # [N, d_v]
            h_scalar = torch.cat([sc, V_norm], dim=-1)  # final scalar input for GINE

            # Vector to scalar conversion using L2-norm and fixed W-projection
            # W = torch.tensor([
            #     [ 1,  0,  0],
            #     [-1,  0,  0],
            #     [ 0,  1,  0],
            #     [ 0, -1,  0],
            #     [ 0,  0,  1],
            #     [ 0,  0, -1],
            # ], dtype=torch.float)
            # # V_norm = torch.norm(Ve, dim=-1, keepdim=True)
            # V_norm = torch.norm(Ve, dim=-1)
            # V_proj = torch.matmul(Ve, W.T)
            # V_proj = V_proj.view(Ve.size(0), -1) 
            # V_features = torch.cat([V_norm, V_proj], dim=-1)
            # h_scalar = torch.cat([sc, V_features], dim=-1)

            # Vector to scalar conversion using dot and cross product of the two vectors in Ve
            # V_norm = torch.norm(Ve, dim=-1)   # [N, d_v]
            # v0, v1 = Ve.unbind(dim=1)  # [N, 3], [N, 3]
            # dot = (v0 * v1).sum(dim=-1, keepdim=True)  # [N, 1]
            # cross = torch.cross(v0, v1, dim=-1)        # [N, 3]
            # cross_norm = torch.norm(cross, dim=-1, keepdim=True)  # [N, 1]
            # V_features = torch.cat([V_norm, dot, cross_norm], dim=-1)
            # h_scalar = torch.cat([sc, V_features], dim=-1)

            # Combine L2 norm, Projection, Dot and Cross product
            # V_features = torch.cat([V_norm, V_proj, dot, cross_norm], dim=-1)
            # h_scalar = torch.cat([sc, V_features], dim=-1)

            return h_scalar
########### GVP Model run ########### 



###### Masked node embeddings for GVP node scalar features ######
def mask_residue_embeddings(residue_embeddings, mask_ratio=0.4):
    N = residue_embeddings.shape[0]

    # 1. Sample random mask
    num_mask = max(1, round(mask_ratio * N))
    random_index = torch.randperm(N, device=device)
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[random_index[:num_mask]] = True

    # 2. Clone to avoid in-place issues
    residue_embeddings_masked = residue_embeddings.clone()

    # 3. Mask nucleotide features (assuming first 4 dims = one-hot)
    residue_embeddings_masked[mask, :4] = 0.0

    return residue_embeddings_masked, mask



###### Masked Edge index ######
def mask_edges(edge_index, edge_attr, mask_ratio):

    E = edge_index.size(1)
    num_mask = int(mask_ratio * E)
    perm = torch.randperm(E)
    
    masked_idx = perm[:num_mask]
    unmasked_idx = perm[num_mask:]
   
    unmasked_edge_index = edge_index[:, unmasked_idx]
    masked_edge_index = edge_index[:, masked_idx]
    unmasked_edge_attr = edge_attr[unmasked_idx]
    masked_edge_attr = edge_attr[masked_idx]
   
    return unmasked_edge_index, masked_edge_index, unmasked_edge_attr, masked_edge_attr



def mask_edges_bidirectional(edge_index, edge_attr, mask_ratio):
    """
    Randomly masks a fraction of edges and their reverse counterparts (u->v, v->u).
    Works for directed or undirected graphs.
    """
    E = edge_index.size(1)
    num_mask = int(mask_ratio * E)

    if num_mask == 0:
        # Nothing to mask
        return edge_index, torch.empty((2,0), dtype=torch.long), edge_attr, torch.empty((0, edge_attr.size(1)))

    perm = torch.randperm(E, device=edge_index.device)
    masked_idx = perm[:num_mask]

    src = edge_index[0, masked_idx]
    dst = edge_index[1, masked_idx]

    # Identify reverse edges
    rev_mask_idx = []
    for i in range(len(src)):
        s, d = src[i], dst[i]
        rev = ((edge_index[0] == d) & (edge_index[1] == s)).nonzero(as_tuple=False).view(-1)
        if len(rev) > 0:
            rev_mask_idx.append(rev)
    
    if len(rev_mask_idx) > 0:
        rev_mask_idx = torch.cat(rev_mask_idx)
        all_masked_idx = torch.unique(torch.cat([masked_idx, rev_mask_idx]))
    else:
        all_masked_idx = masked_idx

    # Boolean mask for unmasked edges
    mask = torch.ones(E, dtype=torch.bool, device=edge_index.device)
    mask[all_masked_idx] = False

    unmasked_edge_index = edge_index[:, mask]
    masked_edge_index = edge_index[:, ~mask]
    unmasked_edge_attr = edge_attr[mask]
    masked_edge_attr = edge_attr[~mask]

    return unmasked_edge_index, masked_edge_index, unmasked_edge_attr, masked_edge_attr


def GVP_model_train(batch, gvp_model, optimizer_gvp, gvp_criterion):
    
    batch = batch.to(device)
    gvp_model.train()
    optimizer_gvp.zero_grad()
    
    gvp_x = batch.gvp_x_dict["nucleotide"]
    gvp_edge_attr = batch.gvp_edge_attr_dict["nucleotide"]
    seq = batch.seq_dict["nucleotide"]
    batch_dict = batch.batch_dict["nucleotide"]
    distance_edge_index = batch.edge_index_dict[('nucleotide', 'distance_from', 'nucleotide')]
    mask = batch.mask_dict["nucleotide"]

    num_nodes = int(mask.sum())
    logits = gvp_model(gvp_x, distance_edge_index, gvp_edge_attr, seq, True)
    logits, seq = logits[mask], seq[mask]

    loss_value = gvp_criterion(logits, seq)
    loss_value.backward()
    optimizer_gvp.step()

    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    true = seq.detach().cpu().numpy()
    correct = (pred == true).sum()

    return loss_value, num_nodes, correct
   

def GVP_model_val(batch, gvp_model, gvp_criterion):

    batch = batch.to(device)
    gvp_model.eval()
   
    gvp_x = batch.gvp_x_dict["nucleotide"]
    gvp_edge_attr = batch.gvp_edge_attr_dict["nucleotide"]
    seq = batch.seq_dict["nucleotide"]
    batch_dict = batch.batch_dict["nucleotide"]
    distance_edge_index = batch.edge_index_dict[('nucleotide', 'distance_from', 'nucleotide')]
    mask = batch.mask_dict["nucleotide"]

    num_nodes = int(mask.sum())
    logits = gvp_model(gvp_x, distance_edge_index, gvp_edge_attr, seq, True)
    logits, seq = logits[mask], seq[mask]
    
    loss_value = gvp_criterion(logits, seq)
    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    true = seq.detach().cpu().numpy()
    correct = (pred == true).sum()

    return loss_value, num_nodes, correct


def GVP_model_test(batch, gvp_model, gvp_criterion):
    
    batch = batch.to(device)
    gvp_model.eval()
  
    gvp_x = batch.gvp_x_dict["nucleotide"]
    gvp_edge_attr = batch.gvp_edge_attr_dict["nucleotide"]
    seq = batch.seq_dict["nucleotide"]
    batch_dict = batch.batch_dict["nucleotide"]
    distance_edge_index = batch.edge_index_dict[('nucleotide', 'distance_from', 'nucleotide')]
    mask = batch.mask_dict["nucleotide"]

    num_nodes = int(mask.sum())
    logits = gvp_model(gvp_x, distance_edge_index, gvp_edge_attr, seq, True)
    logits, seq = logits[mask], seq[mask]
    
    loss_value = gvp_criterion(logits, seq)
    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    true = seq.detach().cpu().numpy()
    correct = (pred == true).sum()

    return loss_value, num_nodes, correct
    
    

def autoencoder_train(batch, gvp_model, model, optimizer):

    batch = batch.to(device)
    
    ### Running GVPConv model
    gvp_x = batch.gvp_x_dict["nucleotide"]
    gvp_edge_attr = batch.gvp_edge_attr_dict["nucleotide"]
    seq = batch.seq_dict["nucleotide"]
    batch_dict = batch.batch_dict["nucleotide"]
    distance_edge_index = batch.edge_index_dict[('nucleotide', 'distance_from', 'nucleotide')]
    output = gvp_model(gvp_x, distance_edge_index, gvp_edge_attr, seq, False)
   
    ### Merging GVP model output (nt features) to existing nt features
    nt_dict = batch.x_dict["nucleotide"]
    merged_dict = torch.cat([nt_dict, output], dim=1)
    merged_x_dict = {"nucleotide": merged_dict}
    
    model.train()
    optimizer.zero_grad()
    x_dict = batch.x_dict
    edge_index_dict = batch.edge_index_dict
    edge_attr_dict = batch.edge_attr_dict
    num_nodes = len(x_dict["nucleotide"])
    z_dict = model(merged_x_dict, edge_index_dict, edge_attr_dict)
    
    ### Loss function
    loss = 0
    for edge_type in edge_index_dict:

        if edge_type == ('nucleotide', 'distance_from', 'nucleotide'):
            continue

        pos_src = edge_index_dict[edge_type][0]
        pos_dst = edge_index_dict[edge_type][1]
        pos_pred = (z_dict[edge_type[0]][pos_src] * z_dict[edge_type[2]][pos_dst]).sum(dim=1) # Decoder

        # Negative edges (sampled randomly)
        num_pos = pos_src.size(0)
        neg_edge_index_dict = negative_sampling(edge_index_dict[edge_type], num_nodes=num_nodes, num_neg_samples=num_pos)
        neg_src = neg_edge_index_dict[0]
        neg_dst = neg_edge_index_dict[1]
        neg_pred = (z_dict[edge_type[0]][neg_src] * z_dict[edge_type[2]][neg_dst]).sum(dim=1) # Decoder
        
        pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        loss += pos_loss + neg_loss
     
    loss = loss / (len(edge_index_dict)-1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def autoencoder_val(batch, gvp_model, model):

    batch = batch.to(device)

    ### Running GVPConv model
    gvp_x = batch.gvp_x_dict["nucleotide"]
    gvp_edge_attr = batch.gvp_edge_attr_dict["nucleotide"]
    seq = batch.seq_dict["nucleotide"]
    batch_dict = batch.batch_dict["nucleotide"]
    distance_edge_index = batch.edge_index_dict[('nucleotide', 'distance_from', 'nucleotide')]
    output = gvp_model(gvp_x, distance_edge_index, gvp_edge_attr, seq, False)
    
    ### Merging GVP model output (nt features) to existing nt features
    nt_dict = batch.x_dict["nucleotide"]
    merged_dict = torch.cat([nt_dict, output], dim=1)
    merged_x_dict = {"nucleotide": merged_dict}
    
    model.eval()
    x_dict = batch.x_dict
    edge_index_dict = batch.edge_index_dict
    edge_attr_dict = batch.edge_attr_dict
    num_nodes = len(x_dict["nucleotide"])    
    z_dict = model(merged_x_dict, edge_index_dict, edge_attr_dict)

    ### Loss function
    loss = 0
    auc = 0
    for edge_type in edge_index_dict:
      
        if edge_type == ('nucleotide', 'distance_from', 'nucleotide'):
            continue
               
        pos_src = edge_index_dict[edge_type][0]
        pos_dst = edge_index_dict[edge_type][1]
        pos_pred = (z_dict[edge_type[0]][pos_src] * z_dict[edge_type[2]][pos_dst]).sum(dim=1)
        
        # Negative edges (sampled randomly)
        num_pos = pos_src.size(0)
        neg_edge_index_dict = negative_sampling(edge_index_dict[edge_type], num_nodes=num_nodes, num_neg_samples=num_pos)
        neg_src = neg_edge_index_dict[0]
        neg_dst = neg_edge_index_dict[1]
        neg_pred = (z_dict[edge_type[0]][neg_src] * z_dict[edge_type[2]][neg_dst]).sum(dim=1)

        pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        loss += pos_loss + neg_loss
        
        out = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0).cpu()
        pred = torch.sigmoid(out).cpu()
        auc += roc_auc_score(labels, pred)

    auc = auc / (len(edge_index_dict)-1)
    loss = loss / (len(edge_index_dict)-1)
    return loss, auc


@torch.no_grad()
def autoencoder_test(batch, gvp_model, model):

    batch = batch.to(device)

    ### Running GVPConv model
    gvp_x = batch.gvp_x_dict["nucleotide"]
    gvp_edge_attr = batch.gvp_edge_attr_dict["nucleotide"]
    seq = batch.seq_dict["nucleotide"]
    batch_dict = batch.batch_dict["nucleotide"]
    distance_edge_index = batch.edge_index_dict[('nucleotide', 'distance_from', 'nucleotide')]
    output = gvp_model(gvp_x, distance_edge_index, gvp_edge_attr, seq, False)

    ### Merging GVP model output (nt features) to existing nt features
    nt_dict = batch.x_dict["nucleotide"]
    merged_dict = torch.cat([nt_dict, output], dim=1)
    merged_x_dict = {"nucleotide": merged_dict}
    
    model.eval()
    x_dict = batch.x_dict
    edge_index_dict = batch.edge_index_dict
    edge_attr_dict = batch.edge_attr_dict
    num_nodes = len(x_dict["nucleotide"])
    z_dict = model(merged_x_dict, edge_index_dict, edge_attr_dict)

    ### Loss function
    loss = 0
    auc = 0
    tot_label, tot_pred = [], []
    for edge_type in edge_index_dict:
       
        if edge_type == ('nucleotide', 'distance_from', 'nucleotide'):
            continue
                        
        pos_src = edge_index_dict[edge_type][0]
        pos_dst = edge_index_dict[edge_type][1]
        pos_pred = (z_dict[edge_type[0]][pos_src] * z_dict[edge_type[2]][pos_dst]).sum(dim=1)
        
        # Negative edges (sampled randomly)
        num_pos = pos_src.size(0)
        neg_edge_index_dict = negative_sampling(edge_index_dict[edge_type], num_nodes=num_nodes, num_neg_samples=num_pos)
        neg_src = neg_edge_index_dict[0]
        neg_dst = neg_edge_index_dict[1]
        neg_pred = (z_dict[edge_type[0]][neg_src] * z_dict[edge_type[2]][neg_dst]).sum(dim=1)

        pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        loss += pos_loss + neg_loss
        
        out = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0).cpu()
        pred = torch.sigmoid(out).cpu()
        auc += roc_auc_score(labels, pred)

        labels = labels.tolist()
        pred = pred.tolist()
        tot_label = tot_label + labels
        tot_pred = tot_pred + pred

    auc = auc / (len(edge_index_dict)-1)
    loss = loss / (len(edge_index_dict)-1)
    return loss, auc, tot_label, tot_pred




def run_GVP_GINE_model(train_output, train_loader, val_loader, test_loader, gvp_model, optimizer_gvp, gvp_criterion, model, optimizer):

    ########### GVP ###########
    # train_output = "Train_motif_output"
    f_out = open(train_output, "a")
    f_out.write(f'Sarting GVP training...\n')
    f_out.close()
    
    gvp_epochs = 130
    gvp_best_loss = float('inf')
    gvp_no_improve_count = 0
    gvp_patience = 30
    gvp_best_state = None

    ### Train GVP model
    for epoch in range(gvp_epochs):

        f_out = open(train_output, "a")
        batch_Ve_list = []

        total_loss, total_correct, total_count = 0, 0, 0
        for batch in train_loader:
            
            loss_value, numb_nodes, correct = GVP_model_train(batch, gvp_model, optimizer_gvp, gvp_criterion)          
            total_loss += loss_value.detach().item() * numb_nodes
            total_count += numb_nodes
            total_correct += correct
 
        train_loss = total_loss / total_count
        train_correct = total_correct / total_count
        if epoch % 5 == 0: f_out.write(f'EPOCH {epoch} TRAIN loss: {train_loss:.4f} acc: {train_correct:.4f}\n')

        ### Validation
        total_loss, total_correct, total_count = 0, 0, 0
        for batch in val_loader:

            loss_value, numb_nodes, correct = GVP_model_val(batch, gvp_model, gvp_criterion)          
            total_loss += loss_value.detach().item() * numb_nodes
            total_count += numb_nodes
            total_correct += correct

        val_loss = total_loss / total_count
        val_correct = total_correct / total_count
        if epoch % 5 == 0: f_out.write(f'EPOCH {epoch} VAL loss: {val_loss:.4f} acc: {val_correct:.4f}\n')
                            
        if val_loss < gvp_best_loss:
            gvp_best_loss = val_loss
            gvp_no_improve_count = 0
            gvp_best_state = gvp_model.state_dict()
        else:
            gvp_no_improve_count += 1

        # If loss hasn’t improved in 'patience' epochs, stop training
        if gvp_no_improve_count >= gvp_patience:
            f_out.write(f"Stopping early at epoch {epoch + 1}.\n")
            gvp_model.load_state_dict(gvp_best_state)
            break
        gvp_model.load_state_dict(gvp_best_state)

        f_out.close()

    ###### Testing ######           
    total_loss, total_correct, total_count = 0, 0, 0
    for batch in test_loader:

        loss_value, numb_nodes, correct = GVP_model_test(batch, gvp_model, gvp_criterion)          
        total_loss += loss_value.detach().item() * numb_nodes
        total_count += numb_nodes
        total_correct += correct

    test_loss = total_loss / total_count
    test_correct = total_correct / total_count
            
    f_out = open(train_output, "a")
    f_out.write(f'EPOCH {epoch} TEST loss: {test_loss:.4f} acc: {test_correct:.4f}\n')
    f_out.close()



    ########### GINE Autoencoder ###########
    f_out = open(train_output, "a")
    f_out.write(f'Sarting GINE Autoencoder training...\n')
    f_out.close()
    
    epochs = 200
    best_loss = float('inf')
    no_improve_count = 0
    patience = 30
    best_state = None

    for epoch in range(epochs):

        f_out = open(train_output, "a")
        
        ###### Training ######
        tot_loss = 0
        for batch in train_loader:
            loss = autoencoder_train(batch.to(device), gvp_model, model, optimizer)
            tot_loss += loss

        tot_loss = tot_loss / len(train_loader)
        if epoch % 5 == 0: f_out.write(f"Epoch {epoch}: Train Loss = {tot_loss.item():.4f}\n")

        ### Validation
        tot_loss = 0
    
        for batch in val_loader:
            loss, auc = autoencoder_val(batch.to(device), gvp_model, model)
            tot_loss += loss
 
        if epoch % 5 == 0: f_out.write(f"Epoch {epoch}: Validation Loss = {tot_loss.item():.4f}, Val AUC = {auc:.4f}\n")
        
        tot_loss = tot_loss / len(val_loader)
        if tot_loss < best_loss:
            best_loss = tot_loss
            no_improve_count = 0
            best_state = model.state_dict()
        else:
            no_improve_count += 1
        # If loss hasn’t improved in 'patience' epochs, stop training
        if no_improve_count >= patience:
            f_out.write(f"Stopping early at epoch {epoch + 1}.\n")
            model.load_state_dict(best_state)
            break
        model.load_state_dict(best_state)
        
        f_out.close()

    ###### Testing ######           
    test_auc = 0
    test_loss = 0
    tot_label, tot_pred = [], []
    for batch in test_loader:
        
        loss, auc, label, pred = autoencoder_test(batch.to(device), gvp_model, model)
        test_auc += auc
        test_loss += loss
        tot_label = tot_label + label
        tot_pred = tot_pred + pred

    test_auc = test_auc / len(test_loader)
    test_loss = test_loss / len(test_loader)
    
    f_out = open(train_output, "a")
    f_out.write(f"Test AUC: {test_auc:.4f}, Loss = {test_loss:.4f}\n")
    f_out.close()

    ### Save Model weight
    torch.save(gvp_best_state, "model_weight/gvp_best_model_weights.pt")
    torch.save(best_state, "model_weight/best_model_weights.pt")
    
    return test_loss, test_auc, tot_label, tot_pred
    

def run_model(train_data_path, output_path, family_list, save_par, val_percen, test_percen):
    start_time = time.time()

    train_output = output_path + "Train_motif_output"
    f_out = open(train_output, "w")
    f_out.close()

    Graph_dic = {}
    Graph_count = 0
    motif_family_no = len(family_list)
    family_dic = {}
    family_labels = {}
    Graph_id_map = {}
    
    for key in family_list:
        fam_count = 0
        for file in os.listdir(train_data_path):
            if file.startswith(key):
                fam_count += 1
        family_dic[key] = fam_count
    
    for key in family_dic:
        member_no = family_dic[key]

        for i in range(member_no):
            fin = open(train_data_path + key + '_Graph_' + str(i) + '.g')
            num_nodes, node_features, distance_edge_src, distance_edge_des, distance_edge_features, bond_edge_src, bond_edge_des, bond_edge_features, orientation_edge_src, orientation_edge_des, orientation_edge_features, graph_label, PDB_location, h_V, h_E, seq, mask = read_graph(fin)
            Graph_dic[Graph_count] = [num_nodes, node_features, distance_edge_src, distance_edge_des, distance_edge_features, bond_edge_src, bond_edge_des, bond_edge_features, orientation_edge_src, orientation_edge_des, orientation_edge_features, graph_label, PDB_location, h_V, h_E, seq, mask]
            Graph_id = graph_label + "_" + PDB_location
            Graph_id_map[Graph_count] = Graph_id
            Graph_count += 1


    # ### Data Augmentation using Edge Masking ###
    # max_family_member_count = max(family_dic.values())
    # max_count_family = max(family_dic, key=family_dic.get)
    # augment_family_list = copy.deepcopy(family_list)
    # augment_family_list.remove(max_count_family)
    # Augmented_graph_dic = copy.deepcopy(Graph_dic)
    
    # for family in family_dic:
    #     augment_family_count = max_family_member_count - family_dic[family]
    #     current_family_list = []
       
    #     for g in Graph_dic:
    #         if Graph_dic[g][11] == family:
    #             current_family_list.append(g)
        
    #     random_graphs = random.choices(current_family_list, k=augment_family_count)
    #     for g in random_graphs:
    #         temp_graph_dic = copy.deepcopy(Graph_dic[g])
    #         cur_bond_edge_src, cur_bond_edge_des, cur_bond_edge_features = temp_graph_dic[5], temp_graph_dic[6], temp_graph_dic[7]
    #         cur_edge_index = torch.tensor([cur_bond_edge_src, cur_bond_edge_des], dtype=torch.long)            
    #         cur_edge_attr = torch.tensor(cur_bond_edge_features, dtype=torch.float)
    #         unmasked_edge_index, masked_edge_index, unmasked_edge_attr, masked_edge_attr = mask_edges_bidirectional(cur_edge_index, cur_edge_attr, mask_ratio=0.15)
            
    #         unmasked_edge_index = unmasked_edge_index.tolist()
    #         unmasked_bond_edge_src, unmasked_bond_edge_des = unmasked_edge_index[0], unmasked_edge_index[1]
    #         unmasked_bond_edge_features = unmasked_edge_attr.tolist()
    #         temp_graph_dic[5], temp_graph_dic[6], temp_graph_dic[7] = unmasked_bond_edge_src, unmasked_bond_edge_des, unmasked_bond_edge_features

    #         # Augmented_graph_dic[Graph_count] = Graph_dic[g]
    #         Augmented_graph_dic[Graph_count] = temp_graph_dic
    #         Graph_count += 1
    ### Data Augmentation using Edge Masking ###


    f_out = open(train_output, "a")
    dataset = MyDataset(root='GVPGINE_motif_hetero_dt', num_graphs=len(Graph_dic), Graph_dic=Graph_dic)
    metadata = dataset[0].metadata()
    data = dataset[0] #Access a single graph
    f_out.write(f"Number of graphs: {len(dataset)}\n")
    f_out.write(f"Node types: {data.node_types}\n")
    f_out.write(f"Edge types: {data.edge_types}\n")
    f_out.write(f"Author node features shape: {data['nucleotide'].x.shape}\n")
    f_out.close()

    edge_dims = {}
    for edge_type in dataset[0].edge_attr_dict:
        edge_dims[edge_type] = dataset[0].edge_attr_dict[edge_type].shape[1]


    ########### Autoencoder for link prediction (adjacency reconstruction) ###########
    input_node_dim = (dataset[0]['nucleotide'].x.shape[1] + 10) ### Merging new GVP generated nt features
    # input_node_dim = (dataset[0]['nucleotide'].x.shape[1] + 22) ### Norm + Projection accross 6 dimesion
    # input_node_dim = (dataset[0]['nucleotide'].x.shape[1] + 12) ### Norm + dot product + norm of cross product
    # input_node_dim = (dataset[0]['nucleotide'].x.shape[1] + 24) ### Norm + Projection + dot product + norm of cross product    
    
    gvp_criterion = nn.CrossEntropyLoss()
    model = HeteroGraphAutoEncoder(metadata, input_node_dim=input_node_dim, hidden_dim=64, edge_dims=edge_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    node_dim = (8, 2)
    edge_dim = (4, 1)
    gvp_model = GVPModel((4, 1), node_dim, (1, 1), edge_dim).to(device)
    optimizer_gvp = torch.optim.Adam(gvp_model.parameters(), lr=1e-3)

    indices_list = []
    for i in range(1):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices_list.append(indices)
        # f_out = open(train_output, "a")
        # f_out.write(str(indices)+"\n")
        # f_out.close()

    indices = indices_list[0]
    shuffled_dataset = [dataset[i] for i in indices]
    updated_dataset = []
    
    for graph in shuffled_dataset:
        
        for edge_type in graph.edge_types:
            
            num_nodes = len(graph["nucleotide"]['x'])
            num_edges = graph[edge_type]['edge_index'].size(1)
            edge_index = graph[edge_type]['edge_index']
            edge_attr = graph[edge_type]['edge_attr']
            
            unmasked_edge_index, masked_edge_index, unmasked_edge_attr, masked_edge_attr = mask_edges(edge_index, edge_attr, mask_ratio=0.3)
            graph[edge_type].unmasked_edge_index = unmasked_edge_index.long()
            graph[edge_type].unmasked_edge_attr = unmasked_edge_attr
            graph[edge_type].pos_edge_label_index = masked_edge_index.long()
            graph[edge_type].pos_edge_label_attr = masked_edge_attr
            graph[edge_type].pos_edge_label = torch.ones(len(masked_edge_index[0]))

            if edge_type == ('nucleotide', 'distance_from', 'nucleotide'):
                continue

            num_neg_edges = masked_edge_index.size(1)
            neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_edges)
            graph[edge_type].neg_edge_label_index = neg_edge_index.long()
            graph[edge_type].neg_edge_label = torch.zeros(len(neg_edge_index[0]))

        updated_dataset.append(graph)


    ### Splitting Dataset into Train, Val and Test ###

    split_va, split_te = int(float(val_percen) * len(updated_dataset)), int(float(test_percen) * len(updated_dataset))

    batch_size = 8
    # train_dataset = updated_dataset[:464]
    # val_dataset = updated_dataset[464:504]
    # test_dataset = updated_dataset[504:544]
    train_dataset = updated_dataset[:split_va]
    val_dataset = updated_dataset[split_va:split_te]
    test_dataset = updated_dataset[split_te:len(updated_dataset)]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
  
    #### Training and Testing part ####
    Y_TEST = []
    Y_PRED = []
    TOT_LOSS = 0
    TOT_AUC = 0
    RUN_NO = 3
    for ind in range(RUN_NO):
        f_out = open(train_output, "a")
        f_out.write(f"Run no:: {ind+1}\n")
        f_out.close()
       
        loss, auc, tot_label, tot_pred = run_GVP_GINE_model(train_output, train_loader, val_loader, test_loader, gvp_model, optimizer_gvp, gvp_criterion, model, optimizer)
        Y_TEST = Y_TEST + tot_label
        Y_PRED = Y_PRED + tot_pred
        TOT_LOSS += loss
        TOT_AUC += auc

    f_out = open(train_output, "a")
    auc = roc_auc_score(Y_TEST, Y_PRED)
    f_out.write(f"TOT_LOSS: {TOT_LOSS/RUN_NO}\n")
    f_out.write(f"TOT_AUC: {TOT_AUC/RUN_NO}\n")
    f_out.write(f"List AUC: {auc}\n")
    f_out.close()
    #### Training and Testing part ####
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time using time.time(): {elapsed_time:.4f} seconds")

   
    ########### Embedding for clustering ###########
    embeddings = []
    graph_ids = []

    model.eval()
    for batch in test_loader:
        with torch.no_grad():

            batch = batch.to(device)

            ### Running GVPConv model
            gvp_x = batch.gvp_x_dict["nucleotide"]
            gvp_edge_attr = batch.gvp_edge_attr_dict["nucleotide"]
            seq = batch.seq_dict["nucleotide"]
            batch_dict = batch.batch_dict["nucleotide"]
            distance_edge_index = batch.edge_index_dict[('nucleotide', 'distance_from', 'nucleotide')]
            output = gvp_model(gvp_x, distance_edge_index, gvp_edge_attr, seq, False)

            ### Merging GVP model output (nt features) to existing nt features
            nt_dict = batch.x_dict["nucleotide"]
            merged_dict = torch.cat([nt_dict, output], dim=1)
            merged_x_dict = {"nucleotide": merged_dict}
            
            z_dict = model(merged_x_dict, batch.edge_index_dict, batch.edge_attr_dict)

            # Pooling: get a single embedding per graph
            graph_embedding = global_mean_pool(
                z_dict['nucleotide'], batch.batch_dict['nucleotide']
            )  # shape: [batch_size, hidden_dim]


            embeddings.append(graph_embedding)
            graph_id_list = batch.graph_id.tolist()

            for graph in graph_id_list:
                Graph_id = Graph_id_map[graph]
                graph_ids.append(Graph_id)

    embeddings = torch.cat(embeddings, dim=0)

   
    ########### Write features in output file ###########
    output_file = os.path.join(output_path + "Motif_train_embeddings.tsv")
    fcsv = open(output_file, "w")

    channels = 64
    header = ['Motif_id']
    for i in range(1, channels+1):
        feature_head = 'Feature_' + str(i)
        header.append(feature_head)
    header.append('Label')
    header = str(header)
    header = header.strip('[').strip(']').replace("'", "").replace(',', '\t').replace(" ", "")
    fcsv.write("%s\n" % (header))

    for i in range(len(graph_ids)):
        current_embeddings = embeddings[i].tolist()
        current_embeddings = str(current_embeddings)
        current_embeddings = current_embeddings.strip('[').strip(']').replace(',', '\t')
        graph_id = graph_ids[i].split("_")
        # family, pdb, chain = graph_id[0], graph_id[1], graph_id[2] + "_" + graph_id[3]
        family, pdb = graph_id[0], graph_id[1]

        chain = ""
        for ch_i in range(2, len(graph_id)):
            chain = chain + graph_id[ch_i]
            if ch_i != len(graph_id) -1:
                chain = chain + "_"

        pdb_id = pdb + "_" + chain
        fcsv.write("%s\t%s\t%s\n" % (pdb_id, current_embeddings, family))
        
    fcsv.close()
    ########### Autoencoder for link prediction (adjacency reconstruction) ###########



