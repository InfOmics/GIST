

import numpy as np
import functools
from scipy.sparse import issparse
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import pairwise_distances_chunked

from scipy.sparse import issparse
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from collections import defaultdict, deque

def _atol_for_type(dtype):
    """
    Get the absolute tolerance for a given numpy data type.

    Parameters
    ----------
    dtype : np.dtype
        The data type for which to get the machine epsilon.

    Returns
    -------
    float
        The machine epsilon for the given data type.
    """
    return np.finfo(dtype).eps if np.issubdtype(dtype, np.floating) else 0

def check_number_of_labels(n_labels, n_samples):
    """
    Check whether the number of labels is valid for silhouette computation.

    Parameters
    ----------
    n_labels : int
        Number of unique labels.
    n_samples : int
        Total number of samples.

    Raises
    ------
    ValueError
        If n_labels is not between 2 and n_samples - 1.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            f"Number of labels is {n_labels}. Valid values are 2 to n_samples - 1 (inclusive)"
        )

def silhouette_spatial_score(
    X, labels, adata, *, metric="euclidean",is_visium=True, sample_size=None, random_state=None, **kwds
):
    """
    Compute the mean silhouette coefficient with spatial penalty.

    Parameters
    ----------
    X : ndarray or sparse matrix
        Feature matrix or precomputed distance matrix.
    labels : array-like
        Cluster labels.
    adata : AnnData
        Annotated data matrix containing spatial information.
    metric : str, optional
        Metric used for distance computation. Default is 'euclidean'.
    is_visium : bool, optional
        Whether to use Visium grid layout for neighbor finding. Default is True.
    sample_size : int, optional
        Size of random sample to compute the score on. Default is None (use all).
    random_state : int or RandomState, optional
        Random state for sampling. Default is None.

    Returns
    -------
    float
        Mean silhouette score adjusted for spatial connectivity.
    """
    if sample_size is not None:
        X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            X, labels = X[indices].T[indices].T, labels[indices]
        else:
            X, labels = X[indices], labels[indices]
    return np.mean(silhouette_samples(X, labels, adata, metric=metric,is_visium=is_visium, **kwds))


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

def construct_interaction(adata, n_neighbors=6):
    """
    Construct a k-nearest neighbor graph based on spatial coordinates.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with 'spatial' coordinates in `obsm`.
    n_neighbors : int, optional
        Number of nearest neighbors. Default is 6.

    Modifies
    --------
    adata.obsm['graph_neigh'] : np.ndarray
        Binary adjacency matrix indicating neighbor relationships.
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




def compute_penalty(spatial_neigh, labels, spot_idx):
    """
    Compute cluster connectivity penalty based on spatial disconnection.

    Parameters
    ----------
    sp_neigh : np.ndarray
        Binary adjacency matrix of neighborhood graph.
    labels : dict
        Mapping of spot IDs to cluster labels.
    spot_idx : iterable
        List of spot IDs corresponding to the graph rows.

    Returns
    -------
    tuple
        penalty_dict : dict
            Mapping of each spot ID to its penalty (1 - connectivity score).
        mean_penalty : float
            Average penalty across all clusters.
    """
    unique_labels = set(labels.values())
    penalty_dict = {}
    penalties = []

    # Precompute spot index mapping for O(1) lookups instead of O(n) np.where()
    spot_to_index = {spot: idx for idx, spot in enumerate(spot_idx)}

    for label in unique_labels:
        label_spots = [spot for spot, lbl in labels.items() if lbl == label]
        total_spots = len(label_spots)
        if total_spots == 0:
            continue

        visited = set()
        component_sizes = []

        for spot in label_spots:
            if spot in visited:
                continue  

            queue = [spot]
            component = set()
            while queue:
                current_spot = queue.pop(0)
                if current_spot in visited:
                    continue
                visited.add(current_spot)
                component.add(current_spot)

                # Get neighbors efficiently using the precomputed index
                node_idx = spot_to_index[current_spot]  # O(1) lookup instead of O(n)
                neighbors = spot_idx[spatial_neigh[node_idx] > 0]  # Efficient filtering

                for neighbor in neighbors:
                    if neighbor not in visited and labels.get(neighbor) == label:
                        queue.append(neighbor)

            component_sizes.append(len(component))

        penalty_value = sum((size / total_spots) ** 2 for size in component_sizes)
        penalties.append(1 - penalty_value)

        for spot in label_spots:
            penalty_dict[spot] = 1 - penalty_value

    mean_penalty = np.mean(penalties) if penalties else 0
    return penalty_dict, mean_penalty



def _silhouette_reduce(D_chunk, start, labels, label_freqs, adata, is_visium=True):
 
    """
    Helper function to compute intra- and inter-cluster distances for a chunk of data.

    Parameters
    ----------
    D_chunk : array-like or sparse matrix
        Chunk of the pairwise distance matrix.
    start : int
        Start index of the chunk in the full dataset.
    labels : np.ndarray
        Array of integer labels.
    label_freqs : np.ndarray
        Frequencies of each label in the full dataset.
    adata : AnnData
        Annotated data with spatial information.
    is_visium : bool, optional
        Whether to use Visium spatial layout. Default is True.

    Returns
    -------
    tuple
        intra_cluster_distances : np.ndarray
            Distances within the same cluster.
        inter_cluster_distances : np.ndarray
            Distances to the closest different cluster.
    """
    n_chunk_samples = D_chunk.shape[0]
    cluster_distances = np.zeros((n_chunk_samples, len(label_freqs)), dtype=np.float32)
    spot_ids = list(adata.obs.index)

    if is_visium and 'array_row' in adata.obs and len(adata.obs['array_row']) and 'array_col' in adata.obs and len(adata.obs['array_col']) :
        sp_neighs = get_sp_neighs(adata)
        n_spots = len(spot_ids)
        interaction = np.zeros((n_spots, n_spots), dtype=np.uint8)
        spot_index = {spot: idx for idx, spot in enumerate(spot_ids)}

        for spot, neighbors in sp_neighs.items():
            for neighbor in neighbors:
                interaction[spot_index[spot], spot_index[neighbor]] = 1

        adata.obsm['spatial_neigh'] = interaction
    else:
        construct_interaction(adata, n_neighbors=6)

    label_dict = {spot_ids[i]: labels[i] for i in range(len(spot_ids))}
    penalties, mean_penalty = compute_penalty(adata.obsm['spatial_neigh'], label_dict, adata.obs.index)
    adata.uns['average_penalty'] = mean_penalty

    if issparse(D_chunk):
        D_chunk = D_chunk.tocsr() if D_chunk.format != "csr" else D_chunk

        for i in range(n_chunk_samples):
            sample_weights = D_chunk.data[D_chunk.indptr[i]:D_chunk.indptr[i + 1]]
            sample_labels = labels[D_chunk.indices[D_chunk.indptr[i]:D_chunk.indptr[i + 1]]]
            spot_id = spot_ids[start + i]
            spot_label = labels[start + i]
            penalty = penalties.get(spot_id, 1.0)

            cluster_distances[i] += np.bincount(sample_labels, weights=sample_weights, minlength=len(label_freqs))
            cluster_distances[i, spot_label] += cluster_distances[i, spot_label] * penalty

    else:
        for i in range(n_chunk_samples):
            sample_weights = D_chunk[i]
            sample_labels = labels
            spot_id = spot_ids[start + i]
            spot_label = labels[start + i]
            penalty = penalties.get(spot_id, 1.0)

            cluster_distances[i] += np.bincount(sample_labels, weights=sample_weights, minlength=len(label_freqs))
            cluster_distances[i, spot_label] += cluster_distances[i, spot_label] * penalty

    end = start + n_chunk_samples
    intra_index = (np.arange(n_chunk_samples), labels[start:end])
    intra_cluster_distances = cluster_distances[intra_index]
    cluster_distances[intra_index] = np.inf
    cluster_distances /= label_freqs
    inter_cluster_distances = cluster_distances.min(axis=1)

    return intra_cluster_distances, inter_cluster_distances

def silhouette_samples(X, labels, adata, *, metric="euclidean", is_visium=True, **kwds):
    """
    Compute the silhouette coefficient for each sample with spatial penalties.

    Parameters
    ----------
    X : array-like or sparse matrix
        Feature matrix or precomputed distance matrix.
    labels : array-like
        Cluster labels.
    adata : AnnData
        Annotated data object with spatial metadata.
    metric : str, optional
        Distance metric to use. Default is 'euclidean'.
    is_visium : bool, optional
        Whether to use Visium spatial layout. Default is True.
    **kwds : dict
        Additional keyword arguments passed to `pairwise_distances_chunked`.

    Returns
    -------
    np.ndarray
        Silhouette coefficient for each sample.
    """
    if metric == "precomputed":
        error_msg = ValueError(
            "The precomputed distance matrix contains non-zero "
            "elements on the diagonal. Use np.fill_diagonal(X, 0)."
        )
        if X.dtype.kind == "f":
            atol = _atol_for_type(X.dtype)
            if np.any(np.abs(X.diagonal()) > atol):
                raise error_msg
        elif np.any(X.diagonal() != 0):  
            raise error_msg

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)

    kwds["metric"] = metric
    reduce_func = functools.partial(
        _silhouette_reduce, labels=labels, label_freqs=label_freqs, adata=adata, is_visium=is_visium
    )
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    denom = (label_freqs - 1).take(labels, mode="clip")
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)

    return np.nan_to_num(sil_samples)


