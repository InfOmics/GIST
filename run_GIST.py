  
import argparse
from GIST.GIST import GIST
import torch
from prepare_ground_truth import *
from GIST.utils.clustering import clusters_n_plot
device =  "cuda" if torch.cuda.is_available() else "cpu"



def get_cluster_size(data_name):
    n_cluster=1
    plot_size=0
    if data_name.split("_")[1] in [ '151669', '151670', '151671', '151672']:
        n_cluster= 5
    elif data_name.split("_")[1] in  [ '151507','151508','151509','151510', '151673', '151674', '151675', '151676']:
         n_cluster= 7
    elif "Human_Breast_Cancer" in data_name :
        n_cluster=20
    elif "Mouse_Brain_Anterior" in data_name: 
        n_cluster=52
    elif "Human_Ovarian_Cancer" in data_name: 
        n_cluster=8
    elif "Mouse_Hippocampus" in data_name: 
         n_cluster=14
         plot_size=35
    elif "Olfactory_Bulb" in data_name: 
         n_cluster=7
         plot_size=35
    elif "Mouse_Visual_Cortex" in data_name: 
         n_cluster=7
         plot_size=250
    elif "Human_Lymph_Node" in data_name: 
         n_cluster=8
    elif "Mouse_Kidney" in data_name: 
         n_cluster=7
    elif "Mouse_Brain" in data_name: 
         n_cluster=11
         plot_size=200
    elif "Axolotl_Brain" in data_name: 
         n_cluster=16
         plot_size=35
    return n_cluster, plot_size

adata=get_adata()
seed=35
GISTModel=GIST(adata=adata, device=device, random_seed=seed, is_visium='Visium')
adata=GISTModel.train()
data_name='DLPFC_151673'    
adata.write_h5ad(f"inputs/Preprocessed/{data_name}.h5ad") 
n_cluster, plot_size=get_cluster_size(data_name)
clusters_n_plot(adata, adata.obsm["DGSI"], f"outputs/{data_name}.png", n_cluster, refinement=True,plot_size=plot_size,seed=seed, is_visium=GISTModel.is_visium)

""" from mclustpy import mclustpy
from sklearn.decomposition import PCA
def mClust (X , n_clusters=7):     
    pca = PCA(n_components=20, random_state=42) 
    data = pca.fit_transform(X)
    res = mclustpy(data, G=n_clusters, modelNames='EEE', random_seed=222)
    mclust_res =np.array(res['classification']).astype(int)
    print(np.unique(mclust_res))
    return mclust_res

import scanpy as sc
adata = sc.read_h5ad(f"inputs/Preprocessed/DLPFC_151673.h5ad")
data_name='DLPFC_151673'
n_cluster, plot_size=get_cluster_size(data_name)
#mClust (adata.obsm["DGSI"] , n_clusters=7)
clusters_n_plot(adata, adata.obsm["DGSI"], f"outputs/{data_name}.png", n_cluster, refinement=True,plot_size=plot_size,  is_visium=True) """


    


""" if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" Create the graph for BertWalkST.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--data_name", type=str, default="DLPFC_151673", help="Data name to distiguish graphs of the different dataset.")
    parser.add_argument("--is_visium", type=int, default=1, help="Data technology is visium or not. Default is visium 1")
    parser.add_argument("--is_h5ad", type=int, default=0, help="Data is scanpy h5ad or not. Default is not 0")
    parser.add_argument("--data_path", type=str, default="inputs/spatial_data/Data/1.DLPFC/151673", help="full path to the dataset.")
    #parser.add_argument("--use_svg", type=int, default=0, help="Specify if to use Spatially variable genes or not")
    parser.add_argument("--seed", type=int, default=35, help="Random seed")
    parser.add_argument("--plot_size", type=int, default=0, help="Tissue spatial plot size")
    args= parser.parse_args()
    create_graph(args.data_path, args.data_name, args.is_visium,args.is_h5ad, args.seed , args.plot_size) """
