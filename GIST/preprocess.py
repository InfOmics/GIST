
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse
from .utils.utilities import *
import scipy.sparse as sp
from scipy.spatial.distance import cdist


def norm_data(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    
def hvg (adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    return adata.var['highly_variable']



def get_sp_neighs(adata):
    """
    Get spatial neighbors for each spot using Visium hex grid layout.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with 'array_row' and 'array_col' in obs.

    Returns
    -------
    dict
        A dictionary mapping spot IDs to a set of neighbor spot IDs.
    """
    inverse_coords = {(adata.obs['array_row'][ind], adata.obs['array_col'][ind]): ind for ind in adata.obs.index}
    sp_neighs = {ind: set() for ind in adata.obs.index}

    for ind in adata.obs.index:
        ind_row, ind_col =int(adata.obs['array_row'][ind]), int(adata.obs['array_col'][ind])
        neighbors = [
            (ind_row, ind_col - 2), (ind_row, ind_col + 2),
            (ind_row - 1, ind_col - 1), (ind_row - 1, ind_col + 1),
            (ind_row + 1, ind_col - 1), (ind_row + 1, ind_col + 1)
        ]
        sp_neighs[ind] = {inverse_coords[n] for n in neighbors if n in inverse_coords}
    
    return sp_neighs

def construct_sp_interaction(adata, cosine_similarities_normalized, sp_neighs):
    """
    Construct a symmetric spatial interaction matrix using precomputed cosine similarities.

    Fills in pairwise similarities only for spatial neighbors and sets all other
    interactions to zero.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (e.g., from Scanpy or Squidpy) with `obs` attribute
        containing cell/barcode indices.
    cosine_similarities_normalized : np.ndarray
        2D array of shape (n_cells, n_cells) containing normalized cosine similarities
        between cell features.
    sp_neighs : dict
        Dictionary mapping each cell index (as in `adata.obs.index`) to a list of
        spatial neighbors.

    Returns
    -------
    sp_neighs_cossim : pd.DataFrame
        Symmetric DataFrame of shape (n_cells, n_cells) with cosine similarities
        between each cell and its spatial neighbors. Non-neighbor values are set to 0.
    """
    sp_neighs_cossim= pd.DataFrame(index=adata.obs.index, columns=adata.obs.index)
    for key, v in sp_neighs.items():
        for val in v:
      
            if pd.isna(sp_neighs_cossim.loc[key, val]):
                similarity = cosine_similarities_normalized[np.where(adata.obs.index==key)[0][0], np.where(adata.obs.index==val)[0][0]]
           
                sp_neighs_cossim.loc[key, val] = similarity
                sp_neighs_cossim.loc[val, key] = similarity
 
    sp_neighs_cossim.fillna(0, inplace=True)
    return sp_neighs_cossim



def get_top_k_neighbors(adata, similarity_df, n_neighbors=3):
    """
    Extract top-k most similar neighbors for each observation and construct a binary
    and weighted adjacency matrix.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the observations (cells/spots).
    similarity_df : pd.DataFrame
        DataFrame of shape (n_cells, n_cells) with similarity scores between observations.
        Only non-zero similarities are considered for neighbor selection.
    n_neighbors : int, optional
        Number of top neighbors to select for each observation (default is 3).

    Returns
    -------
    top_neighbors : dict
        Dictionary mapping each observation index to a list of tuples (neighbor, similarity).
        Only includes the top-k most similar neighbors.

    Notes
    -----
    - Updates `adata.obsm['sp_neigh']` with a binary adjacency matrix (0/1).
    - Updates `adata.obsm['adj']` with a symmetric adjacency matrix where overlapping
      directed edges are thresholded to 1.
    """
    num_obs = adata.n_obs
    interaction = np.zeros((num_obs, num_obs))
    top_neighbors = {}

    obs_index = adata.obs.index.to_numpy()

    for index in similarity_df.index:
        sorted_neighbors = similarity_df.loc[index].drop(index).sort_values(ascending=False)
        selected_neighbors = sorted_neighbors[sorted_neighbors > 0].head(n_neighbors)

        top_neighbors[index] = list(zip(selected_neighbors.index, selected_neighbors.values))

        index_pos = np.where(obs_index == index)[0][0]
        neighbor_positions = [np.where(obs_index == n)[0][0] for n in selected_neighbors.index]

        for pos, sim in zip(neighbor_positions, selected_neighbors.values):
            interaction[index_pos, pos] = sim

    adata.obsm['spatial_neigh'] = (interaction > 0)#.astype(int)
    adj = interaction + interaction.T
    adata.obsm['adj'] = np.where(adj > 1, 1, adj)

    return top_neighbors


def construct_interaction(adata, n_neighbors=3):
    """
    Constructs a spot-to-spot spatial interaction graph using Euclidean distance.

    For each spot in the spatial transcriptomics data, the function computes its
    k-nearest neighbors based on Euclidean distances, creates a binary adjacency matrix
    representing these interactions, and stores it in the `AnnData` object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with `.obsm['spatial']` containing spatial coordinates.
    n_neighbors : int, optional (default: 3)
        Number of nearest neighbors to connect for each spot.

    Returns
    -------
    None
        Updates `adata` in-place with the following:
        - `adata.obsm['distance_matrix']`: Full pairwise Euclidean distance matrix.
        - `adata.obsm['sp_neigh']`: Binary (0/1) k-NN interaction matrix (not symmetric).
        - `adata.obsm['adj']`: Symmetric adjacency matrix constructed by max(interaction, interaction.T).

    Notes
    -----
    This function does not weight the edges — it only records the presence (1) or
    absence (0) of a spatial connection based on k-nearest neighbors.
    """
    position = adata.obsm['spatial']
    n_spots = position.shape[0]

    # Efficiently compute pairwise Euclidean distances
    # Avoid full dense matrix if too large (can use blockwise method if needed)
    distance_matrix = cdist(position, position, metric='euclidean')
   

    # Get indices of k nearest neighbors (excluding self)
    nearest_neighbors = np.argsort(distance_matrix, axis=1)[:, 1:n_neighbors + 1]

    # Build sparse matrix
    row_indices = np.repeat(np.arange(n_spots), n_neighbors)
    col_indices = nearest_neighbors.flatten()
    data = np.ones(len(row_indices), dtype=np.uint8)

    # Create sparse interaction matrix
    interaction = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_spots, n_spots))

    # Symmetrize to get undirected graph
    adj = interaction.maximum(interaction.T)

    # Store in AnnData as sparse matrices
    adata.obsm['spatial_neigh'] = interaction.toarray()
    adata.obsm['adj'] = adj.toarray() 



def get_graph_weight_normalized(data):
    cosine_similarities=cosine_similarity( data,  data)
    cosine_similarities_normalized = (cosine_similarities + np.abs(np.min(cosine_similarities)))
    return cosine_similarities_normalized



def create_graph(adata, is_visium=1):
    """
    Preprocesses and create the graph from the AnnData object.

    This function performs the following steps:
    - Removes spots with zero neighbors (only for Visium data).
    - Selects highly variable genes and normalizes expression data.
    - Computes PCA-reduced embedding of expression data.
    - Calculates normalized cosine similarity.
    - Builds neighborhood graph using spatial proximity and similarity.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial transcriptomics data.

    is_visium : bool, optional (default: True)
        Whether the input data is from 10X Visium platform.

    random_seed : int, optional (default: 35)
        Random seed for PCA dimensionality reduction.

    Returns
    -------
    adata : AnnData
        Preprocessed AnnData object with spatial and expression-based neighborhood graphs
        stored in `.obsm`.
    """
    

    if is_visium and 'array_row' in adata.obs and len(adata.obs['array_row']) and 'array_col' in adata.obs and len(adata.obs['array_col']) :
        sp_neighs = get_sp_neighs(adata)

        print(f"Spot size before attempting to remove spots with zero neighbors: {adata.n_obs}")
        # Collect all spots with no neighbors
        zero_neighbor_spots = [key for key, v in sp_neighs.items() if len(v) == 0]

        if zero_neighbor_spots:
            print(f"Removing {len(zero_neighbor_spots)} spots with no neighbors: {zero_neighbor_spots}")
            adata = adata[~adata.obs_names.isin(zero_neighbor_spots)].copy()

        # Confirm removal
        print(f"Spot size after removal: {adata.n_obs}")

    if 'highly_variable' not in adata.var:
        hvg_genes=hvg(adata)
        norm_data(adata)
        adata= adata[:,hvg_genes]


    if issparse(adata.X):
            data=pca ( adata.X.toarray(), n_components=20) 
    else:
            data=pca ( adata.X, n_components=20) #if data already a matrix

    adata.obsm["X_pca"]=data
    cosine_similarities_normalized=get_graph_weight_normalized(data)

    print("get neighbourhood graph")
    if is_visium and 'array_row' in adata.obs and len(adata.obs['array_row']) and 'array_col' in adata.obs and len(adata.obs['array_col']) :   
        similarity_df=construct_sp_interaction(adata, cosine_similarities_normalized, sp_neighs)
        get_top_k_neighbors(adata, similarity_df, n_neighbors=3)  
    else: construct_interaction(adata, n_neighbors=3)
    print("adata shape ", adata.shape)
    print("Preparing adata for training: Done")
    return adata

 


 
