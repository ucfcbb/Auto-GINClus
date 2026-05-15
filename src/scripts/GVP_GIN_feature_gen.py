import os
import sys
import ast
import torch
import torch.nn as nn
import time
import random

from torch_geometric.data import InMemoryDataset, HeteroData
from torch_geometric.loader import DataLoader
from torch.nn import Sequential, Linear, ReLU, Softplus
from torch_geometric.nn import global_mean_pool

### For GINEConv model
from torch_geometric.nn import GINEConv, HeteroConv
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

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
    h_V = (residue_embeddings, node_coordinates)
    ### Edge features (scalar, vector)
    h_E = (scalar_edge_features, edge_orientations)
    ### Sequence
    letter_to_num = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    seq = torch.as_tensor([letter_to_num[a] for a in SEQ], device=device, dtype=torch.long)
    # mask = torch.isfinite(node_coordinates.sum(dim=(1,2)))
    valid_node_mask = torch.isfinite(node_coordinates.sum(dim=(1,2))) ### Removes nodes with NaN or InF coordinate
    mask = node_mask & valid_node_mask # combining both masks

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
    '''
    GVP-GNN 
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 20 amino acids at each position in a `torch.Tensor` of 
    shape [n_nodes, 20].
    
    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers in each of the encoder
                       and decoder modules
    :param drop_rate: rate to use in all dropout layers
    '''
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
        
        self.W_s = nn.Embedding(4, 4)
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
            V_norm = torch.norm(Ve, dim=-1)   # [N, d_v]
            h_scalar = torch.cat([sc, V_norm], dim=-1)  # final scalar input for GINE

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
def mask_edges(edge_index, edge_attr, mask_ratio=0.3):

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




def run_GVP_GINE_model(dataset, gvp_model, optimizer_gvp, gvp_criterion, model, optimizer, output_file_name):

    ### Masking Dataset and Generating Pos, Neg indexes ###
    updated_dataset = []
    for graph in dataset:
        
        for edge_type in graph.edge_types:
            
            num_nodes = len(graph["nucleotide"]['x'])
            num_edges = graph[edge_type]['edge_index'].size(1)
            edge_index = graph[edge_type]['edge_index']
            edge_attr = graph[edge_type]['edge_attr']
            
            unmasked_edge_index, masked_edge_index, unmasked_edge_attr, masked_edge_attr = mask_edges(edge_index, edge_attr, mask_ratio=0.15)
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

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    ########### GVP ###########
    f_out = open(output_file_name, "a")
    f_out.write(f'Sarting GVP testing...\n')
    f_out.close()
    
    ###### Load GVP model weight ######
    gvp_model.load_state_dict(torch.load("model_weight/gvp_best_model_weights.pt", map_location=device))
    gvp_model.to(device)
        
    ###### Testing GVP model ######           
    total_loss, total_correct, total_count = 0, 0, 0
    for batch in loader:
        loss_value, numb_nodes, correct = GVP_model_test(batch, gvp_model, gvp_criterion)          
        total_loss += loss_value.detach().item() * numb_nodes
        total_count += numb_nodes
        total_correct += correct

    test_loss = total_loss / total_count
    test_correct = total_correct / total_count
            
    f_out = open(output_file_name, "a")
    f_out.write(f'TEST loss: {test_loss:.4f} acc: {test_correct:.4f}\n')
    f_out.close()



    ########### GINE Autoencoder ###########
    f_out = open(output_file_name, "a")
    f_out.write(f'Sarting GINE Autoencoder testing...\n')
    f_out.close()


    ###### Load GINE model weight ######
    model.load_state_dict(torch.load("model_weight/best_model_weights.pt", map_location=device))
    model.to(device)

    ###### Testing GINE model ######           
    test_auc = 0
    test_loss = 0
    tot_label, tot_pred = [], []
    for batch in loader:      
        loss, auc, label, pred = autoencoder_test(batch.to(device), gvp_model, model)
        test_auc += auc
        test_loss += loss
        tot_label = tot_label + label
        tot_pred = tot_pred + pred

    test_auc = test_auc / len(loader)
    test_loss = test_loss / len(loader)
    
    f_out = open(output_file_name, "a")
    f_out.write(f"Test AUC: {test_auc:.4f}, Loss = {test_loss:.4f}\n")
    f_out.close()

    return test_loss, test_auc, tot_label, tot_pred, loader
    


def generate_feature(candidate_data_path, family_list, output_path, input_index_type):

    output_file_name = output_path + "Candidate_motif_output.txt"
    start_time = time.time()

    f_out = open(output_file_name, "w")
    f_out.close()

    Graph_dic = {}
    Graph_count = 0
    motif_family_no = len(family_list)
    family_dic = {}
    family_labels = {}
    Graph_id_map = {}
    

    for key in family_list:
        fam_count = 0
        for file in os.listdir(candidate_data_path):
            if file.startswith(key):
                fam_count += 1
        family_dic[key] = fam_count
    
    for key in family_dic:
        member_no = family_dic[key]

        for i in range(member_no):
            fin = open(candidate_data_path + key + '_Graph_' + str(i) + '.g')
            # print(i)
            num_nodes, node_features, distance_edge_src, distance_edge_des, distance_edge_features, bond_edge_src, bond_edge_des, bond_edge_features, orientation_edge_src, orientation_edge_des, orientation_edge_features, graph_label, PDB_location, h_V, h_E, seq, mask = read_graph(fin)
            Graph_dic[Graph_count] = [num_nodes, node_features, distance_edge_src, distance_edge_des, distance_edge_features, bond_edge_src, bond_edge_des, bond_edge_features, orientation_edge_src, orientation_edge_des, orientation_edge_features, graph_label, PDB_location, h_V, h_E, seq, mask]
            Graph_id = graph_label + "_" + PDB_location
            Graph_id_map[Graph_count] = Graph_id
            Graph_count += 1

    f_out = open(output_file_name, "a")
    dataset = MyDataset(root='GVPGINE_loop_hetero_dt', num_graphs=Graph_count, Graph_dic=Graph_dic)
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
    gvp_criterion = nn.CrossEntropyLoss()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGraphAutoEncoder(metadata, input_node_dim=input_node_dim, hidden_dim=64, edge_dims=edge_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    node_dim = (8, 2)
    edge_dim = (4, 1)
    gvp_model = GVPModel((4, 1), node_dim, (1, 1), edge_dim).to(device)
    optimizer_gvp = torch.optim.Adam(gvp_model.parameters(), lr=1e-3)

    loss, auc, tot_label, tot_pred, loader = run_GVP_GINE_model(dataset, gvp_model, optimizer_gvp, gvp_criterion, model, optimizer, output_file_name)
    f_out = open(output_file_name, "a")
    f_out.write(f"LOSS: {loss}\n")
    f_out.write(f"AUC: {auc}\n")
    f_out.close()

    # loader = DataLoader(dataset, batch_size=8, shuffle=True)

    ########### Embedding for clustering ###########
    embeddings = []
    graph_ids = []

    model.eval()
    for batch in loader:
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
    output_file = os.path.join(output_path, "Motif_candidate_embeddings.tsv")
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Feature generation elapsed time using time.time(): {elapsed_time:.4f} seconds")


