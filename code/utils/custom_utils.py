" custom utils script "

import torch
import pytorch_lightning as pl
import numpy as np
import pickle
import random

# User defined libraries
from nn.vanilla_ae import VanillaAE

def save_to_pickle(file, file_name, save_dir):
    """Saves a file to a pickle file"""
    with open(save_dir + '/' + file_name, 'wb') as f:
        pickle.dump(file, f)

def read_from_pickle(file_name, save_dir):
    """Reads a file from a pickle file"""
    with open(save_dir + '/' + file_name, 'rb') as f:
        file = pickle.load(f)
    return file 

def set_global_random_seed(seed):
    """Sets the global random seed."""

    # Pytorch lightning function to seed RNG for everything
    # Setting workers=True, Lightning derives unique seeds 
    # 1. across all dataloader workers and processes for torch
    # 2. numpy and stdlib random number generators
    pl.seed_everything(seed, workers=True)

    # Seeds RNG for all devices (CPU and GPU)
    torch.manual_seed(seed)

    # Sets the random seed for python
    random.seed(seed)

    # Sets the random seed in numpy library
    np.random.seed(seed)

def create_ae_module(nn_save_dir, nn_params_dict, nn_train_params_dict, nn_datasets_dict):
    """Creates an AutoEncoder module"""
    return VanillaAE(nn_save_dir, nn_params_dict, nn_train_params_dict, nn_datasets_dict)
