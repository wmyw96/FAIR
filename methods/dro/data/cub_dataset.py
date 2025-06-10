import os
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from methods.dro.data.confounder_dataset import ConfounderDataset

class CUBDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self, root_dir,
                 target_name, confounder_names,features,responses,
                 augment_data=False,
                 model_type=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data
        self.env_idx=[responses[0].shape[0],responses[0].shape[0]+responses[1].shape[0],
                      responses[0].shape[0]+responses[1].shape[0]+responses[2].shape[0]]

        


        # Get the y values
        self.y_array =  np.concatenate(responses).flatten()
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = np.zeros(self.y_array.shape[0])
        self.confounder_array[self.env_idx[0]:self.env_idx[1]] = np.ones(responses[1].shape[0])
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        
        self.features_mat = torch.from_numpy(np.concatenate(features)).float()
        self.train_transform = None
        self.eval_transform = None
    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        subsets['train'] = Subset(self,np.arange(0,self.env_idx[1]))
        subsets['val'] = Subset(self,np.arange(self.env_idx[1],self.env_idx[2]))
        subsets['test'] = Subset(self,np.arange(self.env_idx[1],self.env_idx[2]))
        return subsets

