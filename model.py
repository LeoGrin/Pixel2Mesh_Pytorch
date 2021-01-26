import torch.nn as nn
import torch.nn.functional as F
#from pytorch3d.ops import GraphConv
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, GENConv, SGConv
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

from torch_geometric.nn import GENConv, DeepGCNLayer

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(3, 16)
        self.conv2 = GraphConv(16, 3)

    def forward(self, mesh):
        verts = mesh.verts_packed()
        #TODO faster with packed
        edges = mesh.edges_packed()
        out = F.relu(self.conv1(verts, edges))
        return self.conv2(out, edges)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)

        #return F.log_softmax(x, dim=1)



class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.conv_first = GraphConv(in_dim, 128)
        #self.conv1 = GraphConv(16, 16)
        self.conv1 = GraphConv(128, 128) #TODO ?
        self.conv2 = GraphConv(128, 128)
        self.conv3 = GraphConv(128, 128)
        self.conv4 = GraphConv(128, 128)
        self.conv4 = GraphConv(128, 128)
        self.conv5 = GraphConv(128, 128)
        self.conv6 = GraphConv(128, 128)
        self.conv7 = GraphConv(128, 128)
        self.conv8 = GraphConv(128, 128)
        self.conv9 = GraphConv(128, 128)
        self.conv10 = GraphConv(128, 128)
        self.conv11 = GraphConv(128, 128)
        self.conv_last = GraphConv(128, out_dim)
        self.conv_pos = GraphConv(out_dim, 3)
        #self.conv2 = GraphConv(32, 3)
        #self.conv_last =GraphConv(out_dim, 3)
        #self.conv2 = GCNConv(248, 124)
        #self.conv3 = GCNConv(124, 124)
        #self.conv4 = GCNConv(124, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        #print(x.shape)
        #print(self.conv1(x, edge_index).shape)
        x = F.relu(self.conv1(x, edge_index)) + x
        x = F.relu(self.conv2(x, edge_index)) + x
        x = F.relu(self.conv3(x, edge_index)) + x
        x = F.relu(self.conv4(x, edge_index)) + x
        x = F.relu(self.conv5(x, edge_index)) + x
        x = F.relu(self.conv6(x, edge_index)) + x
        x = F.relu(self.conv7(x, edge_index)) + x
        x = F.relu(self.conv8(x, edge_index)) + x
        x = F.relu(self.conv9(x, edge_index)) + x
        x = F.relu(self.conv10(x, edge_index)) + x
        x = F.relu(self.conv11(x, edge_index)) + x
        out_features = x
        #x = F.relu(x) #TODO ?
        out_pos = self.conv_pos(x, edge_index)
        #x = F.relu(x)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)

        return out_features, out_pos#self.conv4(x, edge_index)#F.log_softmax(x, dim=1)

class DeeperGCN(nn.Module):
    def __init__(self, in_dim, hidden_channels, out_dim, num_layers):
        super(DeeperGCN, self).__init__()

        self.node_encoder = nn.Linear(in_dim, hidden_channels)
        #self.edge_encoder = nn.Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            #conv = GraphConv(hidden_channels, hidden_channels)
            #conv = SGConv(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.0, #TODO
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, out_dim)

    def forward(self, data):
        x , edge_index = data.x, data.edge_index
        x = self.node_encoder(x)
        #edge_attr = self.edge_encoder(edge_attr)
        x = self.layers[0].conv(x, edge_index)#, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)#, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        #x = F.dropout(x, p=0.05, training=self.training) #TODO

        return x, self.lin(x)