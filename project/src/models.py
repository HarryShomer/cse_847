import torch
import torch.nn.functional as F
import torch_geometric.nn as conv_layers


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

        self.convs.append(conv_layers.GCNConv(feat_dim, hidden_dim, cached=True))

        for _ in range(num_layers-2):
            self.convs.append(conv_layers.GCNConv(hidden_dim, hidden_dim, cached=True))
        
        self.convs.append(conv_layers.GCNConv(hidden_dim, num_classes, cached=True))


    def forward(self, data):
        """
        Aggregate
        """
        x, edge_index = data.x, data.edge_index

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)

        x = self.convs[-1](x, edge_index)

        return F.log_softmax(x, dim=1)



class GAT(torch.nn.Module):
    """
    Graph Attention Network
    """
    def __init__(self, feat_dim, hidden_dim, num_classes, heads=8, dropout=.6, num_layers=2):
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

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, edge_index)

            # Don't apply for last layer
            if i != len(self.convs) - 1:
                x = F.elu(x)
        
        return F.log_softmax(x, dim=1)



class APPNP(torch.nn.Module):
    """
    APPNP Implementation
    """
    def __init__(self, feat_dim, num_hidden, num_classes, iters, alpha, appnp_drop=.2, mlp_drop=0.1):
        super().__init__()
        self.dropout = appnp_drop

        self.mlp = MLP(feat_dim, num_hidden, num_classes, dropout=mlp_drop)
        self.conv_layer = conv_layers.APPNP(iters, alpha)


    def forward(self, data):
        """
        Aggregate - MLP + Smoothing
        """
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layer(x, edge_index)

        return F.log_softmax(x, dim=1)


class SGC(torch.nn.Module):
    """
    SGC Implementation
    """
    def __init__(self, feat_dim, num_classes, k):
        """
        """
        super().__init__()
        self.conv_layer = conv_layers.SGConv(feat_dim, num_classes, K=k, cached=True)

       
    def forward(self, data):
        """
        Aggregate
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv_layer(x, edge_index)

        return F.log_softmax(x, dim=1)
