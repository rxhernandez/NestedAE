""" Predict script """
import os
import sys
import pickle
import random

import click
from torch import manual_seed, load
import numpy as np
from pytorch_lightning import seed_everything

from NestedAE.ae import AE

def set_global_random_seed(seed):
    """Sets the global random seed."""
    # Pytorch lightning function to seed RNG for everything
    # Setting workers=True, Lightning derives unique seeds 
    # 1. across all dataloader workers and processes for torch
    # 2. numpy and stdlib random number generators
    seed_everything(seed, workers=True)
    # Seeds RNG for all devices (CPU and GPU)
    manual_seed(seed)
    # Sets the random seed for python
    random.seed(seed)
    # Sets the random seed in numpy library
    np.random.seed(seed)

@click.command()
@click.option('--ae_save_dir', prompt='nn', help='Directory where the AE is saved.')
@click.option('--accelerator', prompt='accelerator', help='Specify the type of acceleration to use.')
@click.option('--module', prompt='module', help='Enter the submodule from which to get predictions.')
def run_predict(ae_save_dir, accelerator, module):
    """ Predict script"""
    nn_params_dict = pickle.load(open(ae_save_dir + '/nn_params_dict.pkl', 'rb'))
    nn_train_params_dict = pickle.load(open(ae_save_dir + '/nn_train_params_dict.pkl', 'rb'))
    global_seed = nn_train_params_dict['global_seed']
    # Send all print statements to file for debugging
    print_file_path = ae_save_dir + '/' + 'predict_out.txt'
    sys.stdout = open(print_file_path, "w")
    set_global_random_seed(global_seed)
    # Create a blank ae
    new_ae = AE(ae_save_dir,
                nn_params_dict,
                nn_train_params_dict,
                '../datasets/combined_predict_datasets.pt')
    # Load the weights from the latest checkpoint
    chpt_path = f'{ae_save_dir}/checkpoints/last.ckpt'
    loaded_ae = new_ae.load_from_checkpoint(chpt_path)
    # Call the model on predict  dataset
    predict_dataset_dir = '../datasets/predict_dataset.pt'
    predict_dataset = load(predict_dataset_dir)
    module_outputs_from_loaded = loaded_ae(predict_dataset)
    module_outputs_dir = f'{ae_save_dir}/ae_module_outputs'
    if not os.path.exists(module_outputs_dir + '/predict'):
        os.mkdir(module_outputs_dir + '/predict')
    module_output = module_outputs_from_loaded[module].detach().numpy()
    filename = module + '_output.csv'
    np.savetxt(module_outputs_dir + '/predict/' + filename, module_output, delimiter=',')

if __name__ == '__main__':
    run_predict()
