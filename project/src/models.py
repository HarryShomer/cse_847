import torch
import torch.nn.functional as F
import torch_geometric.nn as conv_layers
# from torch_geometric.nn.models import MLP


class MLP(torch.nn.Module):
    """
    2 Layer MLP
    """
    def __init__(self, in_channels, hid_channels, out_channels, dropout):
        super().__init__()
        self.dropout = dropout

        self.bn = torch.nn.BatchNorm1d(hid_channels)
        self.fc1 = torch.nn.Linear(in_channels, hid_channels)
        self.fc2 = torch.nn.Linear(hid_channels, out_channels)
    

    def forward(self, x):
        """
        Forward Pass
        """
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.fc2(x)

        return x



class GCN(torch.nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, feat_dim, hidden_dim, num_classes, dropout=.2, num_layers=2):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        self.convs.append(conv_layers.GCNConv(feat_dim, hidden_dim))

        for _ in range(num_layers-2):
            self.convs.append(conv_layers.GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(conv_layers.GCNConv(hidden_dim, num_classes))


    def forward(self, data):
        """
        Aggregate
        """
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = F.dropout(x, self.dropout)
            x = F.relu(x)
            x = conv(x, edge_index)

        return x


class GAT(torch.nn.Module):
    """
    Graph Attention Network
    """
    def __init__(self, feat_dim, hidden_dim, num_classes, heads=8, dropout=.2, num_layers=2):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        self.convs.append(conv_layers.GATConv(feat_dim, hidden_dim, heads=heads))

        for _ in range(num_layers-2):
            self.convs.append(conv_layers.GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        
        self.convs.append(conv_layers.GATConv(hidden_dim * heads, num_classes, heads=heads, concat=False))


    def forward(self, data):
        """
        Aggregate
        """
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = F.dropout(x, self.dropout)
            x = F.elu(x)
            x = conv(x, edge_index)

        return x


class APPNP(torch.nn.Module):
    """
    APPNP Implementation
    """
    def __init__(self, feat_dim, num_hidden, num_classes, iters, alpha, appnp_drop=.2, mlp_drop=0.2):
        super().__init__()
        self.dropout = appnp_drop

        self.mlp = MLP(feat_dim, num_hidden, num_classes, dropout=mlp_drop)
        self.conv_layer = conv_layers.APPNP(iters, alpha)


    def forward(self, data):
        """
        Aggregate - MLP + Smoothing
        """
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout)
        x = self.mlp(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.conv_layer(x, edge_index)

        return x
