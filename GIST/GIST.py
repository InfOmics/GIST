import torch
import numpy as np
import torch.nn.functional as F
from GIST.model import GNN
from GIST.preprocess import *
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse import issparse


class GIST() :
    def __init__(self, 
        adata,
        device= torch.device('cpu'),
        learning_rate=0.0005,
        weight_decay=0.001,
        epochs=1100, 
        random_seed = 35,
        is_visium='Visium'
        ):
        self.adata=adata
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.lambda_RL=5
        if is_visium == 'Visium':
            self.is_visium=True
            self.dim_output = 32
            self.dim_hidden = 32
        elif is_visium in ['Slide-seq', 'STARmap', 'Stereo-seq']:
            self.is_visium=False
            self.dim_output =20
            self.dim_hidden =20
        print(f"Using {is_visium} data")

        self.adata=preprocess_adata(self.adata, is_visium=self.is_visium, random_seed=random_seed)
        self.adj = torch.tensor(self.adata.obsm['adj'], dtype=torch.float32, device=self.device)
        self.adj = self.normalize_adj(self.adj).double()         
        
        self.sp_neigh= torch.tensor(self.adata.obsm['sp_neigh'].copy() + np.eye(self.adj.shape[0]), dtype=torch.float64, device=self.device)      
        
        set_seed(self.random_seed)

        if issparse(adata.X):
            self.data = pca(self.adata.X.toarray(),n_components=self.dim_output, random_state=self.random_seed) 
            self.features = torch.tensor(self.adata.X.toarray(), dtype=torch.float64, device=self.device)
        else:
            self.data = pca(self.adata.X,n_components=self.dim_output, random_state=self.random_seed)  #if data already a matrix
            self.features = torch.tensor(self.adata.X, dtype=torch.float64, device=self.device)
        
        self.data= torch.tensor(self.data, dtype=torch.float64, device=self.device) 
        self.dim_input = self.features.shape[1]   

    def normalize_adj(self, adj):
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(adj.sum(1) + 1e-6))
        return D_inv_sqrt @ adj @ D_inv_sqrt 

    def dgi_loss(self, real_scores, fake_scores):
        """
        Computes Jensen-Shannon divergence loss.
        """
        loss_real = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores))  # Log(D(h_real, s))
        loss_fake = F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))  # Log(1 - D(h_fake, s))
    
        return loss_real + loss_fake  # Minimize JS divergence
    
    def graph_Structural_Mutual_infomation(self, Adj, H, beta):
        """
        Compute the loss:
        L = (1/n^2) * sum_ij L_CE(A_ij, W_ij) + beta * (1/n) * sum_i D_KL(W_i || A_i)
    
        Parameters:
            A (torch.Tensor): Adjacency matrix (n x n)
            H (torch.Tensor): Node embeddings (n x d)
            beta (float): Weight for KL divergence term
    
        Returns:
            torch.Tensor: Computed loss value
        """
        A=Adj.clone()
        n = A.size(0)  # Number of nodes
        W= torch.softmax(H @ H.T, dim=-1) 
 
        # Ensure A is a valid probability distribution
        A = A / (A.sum(dim=1, keepdim=True) + 1e-10)

        # Ensure W has no zero values before taking log
        W = W.clamp(min=1e-10)
        
        # Cross-Entropy Loss: L_CE(A_ij, W_ij)
        ce_loss = F.binary_cross_entropy(W, A, reduction='sum') / (n * n)

        # KL Divergence: D_KL(W_i || A_i) for each row (node)
        kl_loss = F.kl_div(W.log(), A, reduction='batchmean')

        # Final loss
        loss = ce_loss  + beta * kl_loss 
    
        return loss

    def gibbs_sample_features(self, x, adj):
        """
        Perform one iteration of Gibbs sampling to generate corrupted node features.

        For each node, its feature vector is replaced with that of a randomly selected
        neighbor. This is used to introduce controlled noise or corruption for contrastive
        or adversarial learning in graph-based models.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape (num_nodes, num_features).
        adj : torch.Tensor
            Binary adjacency matrix of shape (num_nodes, num_nodes), where non-zero
            entries indicate an edge between nodes.

        Returns
        -------
        x_corrupted : torch.Tensor
            Corrupted node feature matrix with the same shape as `x`.
        """
        x_corrupted = x.clone()
        num_nodes = x.size(0)   
        for i in range(num_nodes):
                neighbors = torch.nonzero(adj[i], as_tuple=True)[0]
                if len(neighbors) > 0:
                    random_neighbor = neighbors[torch.randint(len(neighbors), (1,), )]
                    x_corrupted[i] = x[random_neighbor]  
        return x_corrupted


    def train(self):     
        """
        Train the GNN model with graph structural and graph representation information losses.
        Stores the best model (lowest total loss) and outputs final embeddings to AnnData.
    
        Returns
        -------
        AnnData
            Annotated data object with learned embeddings in `obsm["DGSI"]`.
        """  
        self.model = GNN(self.dim_input, self.dim_hidden, self.dim_output).to(self.device).double()
        optimizer = torch.optim.Adam(
                    (param.to(torch.float64) for param in self.model.parameters()),  # Ensure float64
                        lr=self.learning_rate, 
                                weight_decay=self.weight_decay
                                )  

        x_corrupted = None
        best_loss = float("inf")  # Track lowest loss
        best_model_state = None   # Variable to store best model weights

        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            self.model.train()
            optimizer.zero_grad()

            x_corrupted = self.gibbs_sample_features(self.features, self.adj)  # Corrupt features
            emb, pos_score, pos_score_local = self.model(
            self.features, x_corrupted, self.adj, self.sp_neigh, DGSItraining=True
            ) 

            # Compute losses
            recon_loss = F.mse_loss(emb, self.data) 
            loss_SMI = self.graph_Structural_Mutual_infomation(self.adj, emb, beta=1) 
            loss_RMI = self.dgi_loss(pos_score_local[:, 0], pos_score_local[:, 1]) + self.dgi_loss(pos_score[:, 0], pos_score[:, 1])

            loss = self.lambda_RL*recon_loss + loss_RMI +  loss_SMI # -- Very intresting result. DLPFC 151673 ARI. to explore more
            loss.backward()    
            optimizer.step()

            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = self.model.state_dict()
                print(f" Saved Best Model at Epoch {epoch} | Loss: {best_loss:.4f}")
                    
            print(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.4f} | Total Loss: {loss:.4f} | Recon Loss: {recon_loss:.4f} | RMI Loss: {loss_RMI:.4f} | SMI Loss: {loss_SMI:.4f}")

        # Load best model before evaluation
        if best_model_state  is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model with loss: {best_loss:.4f}")
        
        with torch.no_grad():
            self.model.eval()
            emb = self.model(self.features, x_corrupted, self.adj, self.sp_neigh, DGSItraining=False)
            self.adata.obsm["DGSI"] = emb.float().detach().cpu().numpy()
               

        return self.adata
