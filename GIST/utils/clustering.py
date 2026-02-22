
import random
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from sklearn import metrics
from sklearn.metrics import silhouette_score
from .silhouette_spatial import silhouette_spatial_score
from .utilities import pca

import os

from scipy.spatial import *
from sklearn.preprocessing import *

from sklearn.metrics import *
from scipy.spatial.distance import *
from scipy.spatial import distance_matrix
import scipy.sparse as sp
from scipy.spatial.distance import cdist



def refine_label(adata, radius=50, label_key='label'):
    """
    Refine cluster labels based on the most frequent label among spatial neighbors.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial coordinates in `obsm['spatial']`
        and existing cluster labels in `obs[key]`.
    radius : int, optional
        Number of nearest neighbors to consider for label refinement. Default is 50.
    key : str, optional
        Key in `adata.obs` containing the initial cluster labels. Default is 'label'.

    Returns
    -------
    list of str
        Refined labels where each cell is assigned the most common label
        among its spatial neighbors.
    """
    n_neigh = radius
    refined_label = []
    old_type = adata.obs[label_key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance =  cdist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        refined_label.append(max_type)
        
    refined_label = [str(i) for i in list(refined_label)]    
    
    return refined_label

def mclust_clustering(adata,n_pca=20, num_cluster=7,refinement=True, seed=35):
    """
    Perform clustering, optional label refinement

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Requires 'spatial' in `obsm` and optionally 'ground_truth' in `obs`.
    num_cluster : int, optional
        Number of clusters for Mclust. Default is 7.
    refinement : bool, optional
        Whether to apply spatial label refinement. Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is 35.

    Returns
    -------
    adata
        Modifies `adata.obs['cluster']`,
        """

 
    if adata.obsm["GIST_emb"].shape[1] >n_pca:
        data= pca(adata.obsm["GIST_emb"],n_components=n_pca, random_state=seed) 
    else:
          data= adata.obsm["GIST_emb"]
    """ np.random.seed(seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(data), num_cluster, 'EEE') """
    import numpy as np
    np.random.seed(seed)

    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    # Load R library
    robjects.r.library("mclust")

    # Set R random seed
    robjects.r['set.seed'](seed)

    # Access the Mclust function
    rmclust = robjects.r['Mclust']

    # Proper conversion of NumPy array to R object
    with localconverter(robjects.default_converter + numpy2ri.converter):
        r_data = robjects.conversion.py2rpy(data)

    # Call Mclust (e.g., with G=num_cluster and modelNames='EEE')
    res = rmclust(r_data, G=num_cluster, modelNames="EEE")

    mclust_res = np.array(res[-2]).astype(int)
    # Print the first few clusters
    print(np.unique(mclust_res))

    adata.obs["mclust"]=mclust_res.astype(str)

    if refinement:
        adata.obs["cluster"] = refine_label(adata, radius=50, label_key='mclust') 
    else:
       adata.obs["cluster"] = adata.obs["mclust"]   

    return adata


def plot_n_evaluate_cluster(adata, savepath, plot_size=0,  is_visium=True):
    """
    Perform evaluation, and spatial plotting.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Requires 'spatial' in `obsm` and optionally 'ground_truth' in `obs`.

    savepath : str
        Filename to save the spatial plot.
    plot_size : float, optional
        If greater than zero, use `sc.pl.spatial` with given spot size.
        Otherwise, use `sq.pl.spatial_scatter`. Default is 0.
    is_visium : bool, optional
        Whether to assume Visium-style spatial layout for silhouette penalty. Default is True.

    Returns
    -------
    Scores
        Metrics printed include:
        - Adjusted Rand Index (ARI)
        - Adjusted Mutual Information (AMI)
        - Homogeneity
        - Silhouette Score
        - Spatial Silhouette Score with Penalty
    """
    if  'ground_truth' in adata.obs and len(adata.obs['ground_truth']):
      ARI=metrics.adjusted_rand_score( adata.obs['ground_truth'], adata.obs["cluster"] )
      print('ARI:', np.round(ARI, 4))

      # Adjusted Mutual Information (AMI)
      ami = metrics.adjusted_mutual_info_score( adata.obs['ground_truth'], adata.obs["cluster"] )
      print("AMI:", np.round(ami,4))

      # Homogeneity
      homogeneity = metrics.homogeneity_score( adata.obs['ground_truth'], adata.obs["cluster"] )
      print("Homogeneity Score:", np.round(homogeneity,4))

    else: 
       ARI,ami,homogeneity=0.0,0.0,0.0
    
    if len(np.unique(adata.obs["cluster"]))>1:

        silhouette_spatial = silhouette_spatial_score(adata.obsm["X_pca"], adata.obs["cluster"], adata, metric="cosine", is_visium=is_visium) 
        print("silhouette spatial:",np.round(silhouette_spatial,4))
        
        penalty=adata.uns['average_penalty']
        print("SSS average_penalty:",np.round(penalty,4))

        silhouette = silhouette_score(adata.obsm["X_pca"], adata.obs["cluster"], metric='cosine') 
        print("silhouette:",np.round(silhouette,4))

    else: 
        silhouette_spatial,penalty,silhouette, = 0.0,0.0, 0.0
        print("Cluster size is less than 2")

    if plot_size:

      os.makedirs("figures/show/outputs/dgsignn", exist_ok=True)

      sc.pl.spatial(adata, color="cluster", spot_size=plot_size,save=f"/{savepath}") 
      adata.uns.pop('cluster_colors')
    else: 
      
      sq.pl.spatial_scatter(adata, color="cluster",cmap='Paired', save=savepath) 
      adata.uns.pop('cluster_colors')

    return ARI,ami,homogeneity,silhouette_spatial,penalty, silhouette



