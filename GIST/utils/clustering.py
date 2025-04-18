from GIST.preprocess import pca

import ot
import numpy as np
import scanpy as sc
import squidpy as sq
from sklearn import metrics
from .silhouette_spatial import silhouette_spatial_score
from collections import Counter

import os

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
    """ n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    
    return new_type """

    spatial_coords = adata.obsm['spatial']
    original_labels = adata.obs[label_key].values
    distance_matrix = ot.dist(spatial_coords, spatial_coords, metric='euclidean')

    num_cells = distance_matrix.shape[0]
    refined_labels = []

    for i in range(num_cells):
        distances_to_others = distance_matrix[i]
        neighbor_indices = distances_to_others.argsort()[1:radius + 1]  # exclude self

        neighbor_labels = original_labels[neighbor_indices]
        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]

        refined_labels.append(str(most_common_label))

    return refined_labels

def clusters_n_plot(adata, emb, savepath, num_cluster=7, refinement=True, seed=35, plot_size=0,  is_visium=True):
    """
    Perform clustering, optional label refinement, evaluation, and spatial plotting.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Requires 'spatial' in `obsm` and optionally 'ground_truth' in `obs`.
    emb : np.ndarray
        Embedding matrix to use for clustering.
    savepath : str
        Filename to save the spatial plot.
    num_cluster : int, optional
        Number of clusters for Mclust. Default is 7.
    refinement : bool, optional
        Whether to apply spatial label refinement. Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is 35.
    plot_size : float, optional
        If greater than zero, use `sc.pl.spatial` with given spot size.
        Otherwise, use `sq.pl.spatial_scatter`. Default is 0.
    is_visium : bool, optional
        Whether to assume Visium-style spatial layout for silhouette penalty. Default is True.

    Returns
    -------
    None
        Modifies `adata.obs['cluster']`, prints clustering metrics and saves spatial plots.
        Metrics printed include:
        - Adjusted Rand Index (ARI)
        - Adjusted Mutual Information (AMI)
        - Purity Score
        - Homogeneity, Completeness, V-Measure
        - Silhouette Score
        - Spatial Silhouette Score with Penalty
        - Davies-Bouldin Score
    """
 
    data= pca(emb,n_components=20, random_state=seed) 

    np.random.seed(seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(data), num_cluster, 'EEE')
    mclust_res = np.array(res[-2]).astype(int)
    # Print the first few clusters
    print(np.unique(mclust_res))

    adata.obs["mclust"]=np.array( mclust_res).astype(str)

    if refinement:
        adata.obs["cluster"] = refine_label(adata, radius=50, label_key='mclust') 
    else:
       adata.obs["cluster"] = adata.obs["mclust"]   

    if  'ground_truth' in adata.obs and len(adata.obs['ground_truth']):
      ARI=metrics.adjusted_rand_score( adata.obs['ground_truth'], adata.obs["cluster"] )
      print('ARI:', np.round(ARI, 4))

      # Adjusted Mutual Information (AMI)
      ami = metrics.adjusted_mutual_info_score( adata.obs['ground_truth'], adata.obs["cluster"] )
      print("AMI:", np.round(ami,4))

      # Purity Score
      purity = metrics.cluster.contingency_matrix( adata.obs['ground_truth'], adata.obs["cluster"] ).max(axis=1).sum() / len(adata.obs['ground_truth'])
      print("Purity Score:", np.round(purity, 4))

      # Homogeneity
      homogeneity = metrics.homogeneity_score( adata.obs['ground_truth'], adata.obs["cluster"] )
      print("Homogeneity Score:", np.round(homogeneity,4))

      # Completeness
      completeness = metrics.completeness_score( adata.obs['ground_truth'], adata.obs["cluster"] )
      print("Completeness Score:", np.round(completeness,4))

      # V-Measure
      v_measure = metrics.v_measure_score( adata.obs['ground_truth'], adata.obs["cluster"] )
      print("V-Measure Score:", np.round(v_measure, 4))
    else: 
       ARI,ami,purity,homogeneity,completeness,v_measure=0.0,0.0,0.0,0.0,0.0,0.0
    
    if len(adata.obs["cluster"])>1:
        
        silhouette_spatial = silhouette_spatial_score(adata.obsm["X_pca"], adata.obs["cluster"], adata, metric="cosine", is_visium=is_visium)  
        print("silhouette spatial:",np.round(silhouette_spatial,4))
        
        penalty=adata.uns['average_penalty']
        print("SSS average_penalty:",np.round(penalty,4)) 

        silhouette = metrics.silhouette_score(adata.obsm["X_pca"], adata.obs["cluster"], metric='cosine') 
        print("silhouette:",np.round(silhouette,4))

        davies_bouldin=metrics.davies_bouldin_score(adata.obsm["X_pca"], adata.obs["cluster"]) 
        print("davies_bouldin:",np.round(davies_bouldin,4)) 
    else: 
        silhouette_spatial,silhouette,davies_bouldin = 0.0, 0.0, 0.0
        print("Cluster size is less than 2")

    if plot_size:

      os.makedirs("figures/show/outputs/", exist_ok=True)

      sc.pl.spatial(adata, color="cluster", spot_size=plot_size,save=f"/{savepath}") 
      adata.uns.pop('cluster_colors')
    else: 
      
      sq.pl.spatial_scatter(adata, color="cluster",cmap='Paired', save=savepath) 
      adata.uns.pop('cluster_colors')

    #return ARI,ami,purity,homogeneity,completeness,v_measure,silhouette_spatial,penalty, silhouette,davies_bouldin



