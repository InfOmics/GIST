import torch
import os
import numpy as np
import random
from sklearn.decomposition import PCA


def pca (X, n_components=20,random_state=42):

    pca = PCA(n_components, random_state=42) 

    return pca.fit_transform(X)



def set_seed(seed):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ["MKL_CBWR"] = "COMPATIBLE"   
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        



