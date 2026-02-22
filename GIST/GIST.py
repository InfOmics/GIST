import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["MKL_CBWR"] = "COMPATIBLE"   
import torch
#torch.use_deterministic_algorithms(True, warn_only=True)
import numpy as np
import torch.nn.functional as F
from .model import GCN
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from .utils.utilities import *
from scipy.sparse import issparse
from .preprocess import create_graph
from scipy.sparse import coo_matrix


class GIST () :
    def __init__(self, 
        adata,
        device= torch.device('cpu'),
        learning_rate=0.0005,
        weight_decay=0.001,
        epochs= 1100, 
        emb_size=32,
        random_seed = 35,
        very_large_graph=False,       
        data_type='Visium'
        ):
        self.emb_size = emb_size
        self.use_sparse = very_large_graph
       
        if data_type == 'Visium':
            self.is_visium=True
            self.emb_size = 32
            print(f"Using {data_type} dataset")
        else: # ['Slide-seq', 'STARmap', 'Stereo-seq', 'BaristaSeq']:
            self.is_visium=False
            self.emb_size = 20
            self.use_sparse =True

       
           

        self.random_seed = random_seed
   
        set_seed(self.random_seed)
        self.adata=create_graph(adata, is_visium=self.is_visium)
        print(f"Adata after graph creation: {self.adata}")

        
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
       
        
        if issparse(self.adata.X):
            self.features = torch.tensor(self.adata.X.toarray(), dtype=torch.float64, device=self.device)
        else :
            self.features = torch.tensor(self.adata.X, dtype=torch.float64, device=self.device)
        
        if self.use_sparse:
            # Convert to PyTorch sparse tensor
            adj_coo = coo_matrix(self.adata.obsm['adj'])
            indices = np.vstack((adj_coo.row, adj_coo.col))
            values = adj_coo.data
            shape = adj_coo.shape
            self.adj = torch.sparse_coo_tensor(
            torch.tensor(indices, dtype=torch.int64),
            torch.tensor(values, dtype=torch.float64),
            torch.Size(shape),
            device=self.device
            ).coalesce()        
            self.adj = self.normalize_adj_sparse(self.adj)    
            spatial_coo = coo_matrix(self.adata.obsm['spatial_neigh'] + np.eye(self.adj.shape[0]))
            indices = torch.tensor([spatial_coo.row, spatial_coo.col], dtype=torch.int64, device=self.device)
            values  = torch.tensor(spatial_coo.data, dtype=torch.float64, device=self.device)
            self.spatial_neigh = torch.sparse_coo_tensor(indices, values, (self.adj.shape[0], self.adj.shape[0]), device=self.device).coalesce()
        else:
            self.adj = torch.tensor(self.adata.obsm['adj'], dtype=torch.float64, device=self.device)
            self.adj = self.normalize_adj(self.adj)         

            self.spatial_neigh= torch.tensor(self.adata.obsm['spatial_neigh'].copy() + np.eye(self.adj.shape[0]), dtype=torch.float64, device=self.device) 

        self.dim_input = self.features.shape[1]

        

        if issparse(adata.X):
            self.data = pca(self.adata.X.toarray(),n_components=self.emb_size) 
        else:
            self.data = pca(self.adata.X,n_components=self.emb_size)  #if data already a matrix
        
        self.data= torch.tensor(self.data, dtype=torch.float64, device=self.device)
        


    def normalize_adj(self, adj):
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(adj.sum(1) + 1e-6))
        return D_inv_sqrt @ adj @ D_inv_sqrt   
    def normalize_adj_sparse(self, adj):
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()  # Get row sums
        D_inv_sqrt = torch.pow(row_sum + 1e-6, -0.5)  # Compute D^(-1/2)
        D_inv_sqrt = torch.diag(D_inv_sqrt)  # Convert to diagonal matrix
        return torch.sparse.mm(D_inv_sqrt, torch.sparse.mm(adj, D_inv_sqrt))
    
    def  Graph_Contrastive_Learning_loss(self, real_scores, fake_scores):
        """
        Computes Jensen-Shannon divergence loss.
        """
        loss_real = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores))  # Log(D(h_real, s))
        loss_fake = F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))  # Log(1 - D(h_fake, s))
    
        return loss_real + loss_fake  # Minimize JS divergence
    
    def  Graph_Structural_Learning_loss(self, Adj, H, beta):
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
        W= torch.softmax(H@H.T, dim=-1) 
        eps = 1e-10
      
        # Ensure A is a valid probability distribution
        A = A / (A.sum(dim=1, keepdim=True) + eps)

        # Ensure W has no zero values before taking log
        W = W.clamp(min=eps)
        
        # Cross-Entropy Loss: L_CE(A_ij, W_ij)
        ce_loss = F.binary_cross_entropy(W, A, reduction='sum') / (n * n)

        # KL Divergence: D_KL(W_i || A_i) for each row (node)
        kl_loss = F.kl_div(W.log(), A, reduction='batchmean') 

        #print(f"KL Loss: {kl_loss.item()}  || CE Loss: {ce_loss.item()} || Beta: {beta}")
    
        # Final loss
        loss = ce_loss  + beta * kl_loss 
    
        return loss

    def gibbs_sample_features(self, x, adj, num_iters=1):
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

     
        for _ in range(num_iters):
            for i in range(num_nodes):
                neighbors = torch.nonzero(adj[i], as_tuple=True)[0]
                if len(neighbors) > 0:
                    random_idx = torch.randint(len(neighbors), (1,),)
                    random_neighbor = neighbors[random_idx]
                    x_corrupted[i] = x[random_neighbor]

        return x_corrupted


    def train(self):
        """
        Train the GNN model with graph structural and graph representation information losses.
        Stores the best model (lowest total loss) and outputs final embeddings to AnnData.
    
        Returns
        -------
        AnnData
            Annotated data object with learned embeddings in `obsm["GIST_emb"]`.
        """          
        self.model = GCN(self.dim_input, self.emb_size, self.emb_size,self.use_sparse).to(self.device).double()

        optimizer = torch.optim.Adam(
                    (param.to(torch.float64) for param in self.model.parameters()),  # Ensure float64
                        lr=self.learning_rate, 
                                weight_decay=self.weight_decay
                                )   

        x_corrupted = None
        best_loss = float("inf")  # Track lowest loss
        best_model_state = None   # Variable to store best model weights
        lambda_Recon=5
        lambda_GCL= 1
        lambda_GSL= 1
        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            self.model.train()
            optimizer.zero_grad()

            x_corrupted = self.gibbs_sample_features(self.features, self.adj)  # Corrupt features
            emb, pos_score, pos_score_local = self.model(
            self.features, x_corrupted, self.adj, self.spatial_neigh, IsTraining=True
            ) 

            # Compute losses
            recon_loss = F.mse_loss(emb, self.data) 
            loss_GSL = self. Graph_Structural_Learning_loss(self.adj, emb, beta=1) 
            loss_GCL = self. Graph_Contrastive_Learning_loss(pos_score_local[:, 0], pos_score_local[:, 1]) + self. Graph_Contrastive_Learning_loss(pos_score[:, 0], pos_score[:, 1])

            loss = lambda_Recon*recon_loss + lambda_GCL*loss_GCL +  lambda_GSL*loss_GSL 

            loss.backward()
            optimizer.step()   

            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = self.model.state_dict()
                #print(f" Saved Best Model at Epoch {epoch} | Loss: {best_loss:.4f}")

            #print(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.4f} | Total Loss: {loss:.4f} | Recon Loss: {recon_loss:.4f} | RMI Loss: {loss_GCL:.4f} | SMI Loss: {loss_GSL:.4f}")

        # Load best model before evaluation
        if best_model_state  is not None:
            self.model.load_state_dict(best_model_state)
            #print(f"Restored best model with loss: {best_loss:.4f}")
        
        with torch.no_grad():
            self.model.eval()
            emb = self.model(self.features, x_corrupted, self.adj, self.spatial_neigh, IsTraining=False)
            self.adata.obsm["GIST_emb"] = emb.float().detach().cpu().numpy()

        return self.adata
