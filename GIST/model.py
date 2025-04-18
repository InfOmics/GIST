import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Define the GCN Layer
class GCNLayer(nn.Module):
    """
    Graph Convolutional Network (GCN) layer.

    Applies a linear transformation followed by neighborhood aggregation
    using a given adjacency matrix.

    Parameters
    ----------
    in_features : int
        Number of input features per node.
    out_features : int
        Number of output features per node.
    
    Attributes
    ----------
    weight : torch.nn.Parameter
        Learnable weight matrix of shape (in_features, out_features).
    """
    def __init__(self, in_features, out_features ):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(in_features, out_features, dtype=torch.float64))  # Use empty() + float64
        torch.nn.init.xavier_uniform_(self.weight)

        
    def forward(self, features, adj, active=True):
        """
        Forward pass of the GCN layer.

        Parameters
        ----------
        features : torch.Tensor
            Input feature matrix of shape (N, in_features).
        adj : torch.Tensor
            Adjacency matrix of shape (N, N). Should be pre-normalized.
        active : bool, optional
            If True, applies ReLU activation. Default is True.

        Returns
        -------
        torch.Tensor
            Output feature matrix of shape (N, out_features).
        """
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        if active:
            output = F.relu(output)
        return output
    
class Discriminator(nn.Module):
    """
    Discriminator module for contrastive learning.

    This class uses a bilinear scoring function to compute agreement
    between a summary vector (global context) and positive/negative sample embeddings.

    Parameters
    ----------
    n_h : int
        Dimensionality of the hidden embeddings.
    """
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """
        Initialize weights for Bilinear layer using Xavier initialization.
        """
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        """
        Compute bilinear scores between context vector and embeddings.

        Parameters
        ----------
        c : torch.Tensor
            Context vector (global summary).
        h_pl : torch.Tensor
            Positive samples (e.g., real neighbors).
        h_mi : torch.Tensor
            Negative samples (e.g., corrupted features).
        s_bias1 : torch.Tensor, optional
            Bias for positive scores.
        s_bias2 : torch.Tensor, optional
            Bias for negative scores.

        Returns
        -------
        logits : torch.Tensor
            Concatenated logits for positive and negative pairs.
        """
        if c.dim() == 1:
            c_x = torch.unsqueeze(c, 1)
            c_x = c_x.expand_as(h_pl)
        else: 
            c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x) 
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    """
    Average Readout module.

    Computes a summary representation for nodes or neighborhoods
    by averaging feature vectors. If a mask is provided, it performs
    a masked average (e.g., local neighborhood aggregation).

    Methods
    -------
    forward(seq, msk=None)
        Returns the average of the input sequence, optionally masked.
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            """
            Computes local neighborhood summary s_l for each node.

            adj_dense: Dense adjacency matrix (N, N) with self-loops.
            h: Node embeddings (N, hidden_dim).
            """
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk) # Mean of neighbors' embeddings, degree |N_i|
        


class GNN(nn.Module):
    """
    Graph Neural Network with contrastive and local-global representation learning.

    Consists of two GCN layers for both positive and negative views.
    Includes a readout function and a discriminator for contrastive training.

    Parameters
    ----------
    in_dim : int
        Dimension of input features.
    hidden_dim : int
        Dimension of hidden embeddings.
    out_dim : int
        Dimension of output embeddings (projected space).
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GNN, self).__init__()
        
        # GNN layers (positive branch)
        self.conv1 = GCNLayer(in_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, out_dim)

        # GNN layers (negative/corrupted branch)
        self.conv1_neg = GCNLayer(in_dim, hidden_dim)
        self.conv2_neg = GCNLayer(hidden_dim, out_dim)

        # Discriminator for contrastive loss
        self.discriminator = Discriminator(out_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.readout = AvgReadout()

    def forward(self, x, x_neg, edge_index, sp_neigh, DGSItraining=False):
        """
        Forward pass for the GNN.

        Parameters
        ----------
        x : torch.Tensor
            Original node features.
        x_neg : torch.Tensor
            Corrupted node features.
        edge_index : torch.Tensor
            Adjacency matrix (preprocessed).
        sp_neigh : torch.Tensor
            Local neighborhood mask (for spatial or graph structure).
        DGSItraining : bool, optional
            Whether to perform full DGSIGNN training or just forward embedding.

        Returns
        -------
        torch.Tensor or tuple
            If `DGSItraining` is False:
                Returns final node embeddings.
            If `DGSItraining` is True:
                Returns embeddings and discriminator scores for contrastive loss.
        """
        # Standard embedding (no contrastive training)
        if not DGSItraining:
            h = self.conv1(x, edge_index)
            h = self.conv2(h, edge_index, active=False)
            return h

        # Contrastive training mode
        h_pos = self.conv1(x, edge_index)
        z_pos = self.conv2(h_pos, edge_index, active=False)
        z_pos = F.relu(z_pos)

        h_neg = self.conv1_neg(x_neg, edge_index)
        z_neg = self.conv2_neg(h_neg, edge_index)

        # Global summary embedding
        g_global = self.readout(z_pos)
        g_global = self.sigmoid(g_global)

        # Local summary embedding (e.g., spatially-aware)
        g_local = self.readout(z_pos, sp_neigh)
        g_local = self.sigmoid(g_local)

        # Discriminator scores for global and local contrastive loss
        score_global = self.discriminator(g_global, z_pos, z_neg)
        score_local = self.discriminator(g_local, z_pos, z_neg)

        return z_pos, score_global, score_local
    
