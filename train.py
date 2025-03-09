""" Training autoencoder models using wandb sweeps """
import time
import os
import sys
import copy
import random
import pickle

import numpy as np
import click
import wandb
from wandb_api_key import api_key
from torch import load, manual_seed
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

# NestedAE 
from NestedAE.nn_utils import create_callback_object, check_dict_key_exists
from NestedAE.ae import AE

# User input dictionaries
from inputs.nn_inputs import list_of_nn_params_dict
from inputs.train_inputs import list_of_nn_train_params_dict
from inputs.dataset_inputs import list_of_nn_datasets_dict

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

def run_wandb_agent(ae_save_dir_path, nn_params_dict, nn_train_params_dict, nn_datasets_dict, train_dataloader, val_dataloader, accelerator):
    """
    Runs the wandb agent for tuning neural network parameters.

    Args:
        ae_save_dir_path (str): Path to directory that contains all training and model data for the AE.
        nn_params_dict (dict): Dictionary containing the neural network parameters.
        nn_train_params_dict (dict): Dictionary containing the neural network training parameters.
        nn_datasets_dict (dict): Dictionary containing the neural network dataset parameters.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        accelerator (str): Accelerator to use for training.

    Returns:
        None
    """
    # To start a wandb run have to call wandb.init(). At any time there can only be one active run. To finish have to call run.finish()
    wandb.finish()
    run = wandb.init(job_type='training', resume=False, reinit=False, dir='../runs')
    new_nn_params_dict = copy.deepcopy(nn_params_dict)
    for submodule in nn_params_dict['submodules'].keys():
        for submodule_key in nn_params_dict['submodules'][submodule].keys():
            # Set the value chosen by the sweep
            if isinstance(new_nn_params_dict['submodules'][submodule][submodule_key], dict):
                if 'values' in list(new_nn_params_dict['submodules'][submodule][submodule_key].keys()):
                    new_nn_params_dict['submodules'][submodule][submodule_key] = wandb.config[f'{submodule}-{submodule_key}']
            # Get respective params from the encoder and set it for the submodule
            if new_nn_params_dict['submodules'][submodule][submodule_key] == 'mirror':
                new_nn_params_dict['submodules'][submodule][submodule_key] = wandb.config[f'encoder-{submodule_key}']
    
    ae_param_search_path = f'{ae_save_dir_path}/ae_param_search'
    if not os.path.exists(ae_param_search_path):
        os.makedirs(ae_param_search_path)
        print(' --> Created ae_param_search directory.')

    # The directory to store each individual run
    model_sweep_dir = f'{ae_save_dir_path}/ae_param_search/{run.name}'
    # Create a run directory under tune_nn_params
    if not os.path.exists(model_sweep_dir):
        os.makedirs(model_sweep_dir)

    # Send all print statements to file for debugging
    print_file_path = f'{model_sweep_dir}/train_out.txt'
    sys.stdout = open(print_file_path, "w", encoding='utf-8')

    # Build the nn model
    ae = AE(model_sweep_dir, new_nn_params_dict, nn_train_params_dict, f'{ae_save_dir_path}/datasets/combined_train_datasets.pt')
    ae.compile()
    print(' --> Model Compilation step complete.')

    if 'callbacks' in list(nn_train_params_dict.keys()):
        callbacks = create_callback_object(nn_train_params_dict, model_sweep_dir)

    trainer = Trainer(max_epochs=nn_train_params_dict['epochs'],
                      accelerator=accelerator,
                      deterministic=True,
                      logger=WandbLogger(),
                      callbacks=callbacks,
                      enable_model_summary=False,
                      enable_progress_bar=False)
    
    trainer.fit(model=ae, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=None)
    wandb.finish()
    time.sleep(5)
    # Store the input dictionaries to the run directory
    pickle.dump(nn_params_dict, open(f'{model_sweep_dir}/nn_params_dict.pkl', 'wb'))
    pickle.dump(nn_train_params_dict, open(f'{model_sweep_dir}/nn_train_params_dict.pkl', 'wb'))
    pickle.dump(nn_datasets_dict, open(f'{model_sweep_dir}/nn_datasets_dict.pkl', 'wb'))
    # # Move the files from wandb directory to model sweep directory
    # os.system(f'cp -r ../runs/wandb/*-{run.id}/* {model_sweep_dir}')
    # # Delete the run directory from wandb directory
    # os.system(f'rm -r {ae_save_dir}/ae_param_search/wandb/*-{run.id}')
    print(' --> EXIT.')

@click.command()
@click.option('--run_dir', prompt='run_dir',
            help='Specify the run directory.')
@click.option('--ae_save_dir', prompt='ae_save_dir',
               help='Specify the name of directory to save the neural network model.')
@click.option('--ae_idx', prompt='ae_idx',
            help='Specify the neural network to train.')
@click.option('--user_name', prompt='user_name',
            help='Specify the wandb username.')
@click.option('--project_name', prompt='project_name',
            help='Specify the wandb project name.')
@click.option('--sweep_type', prompt='sweep_type',
            help='Specify the sweep type.')
@click.option('--metric', prompt='metric',
            help='Specify the metric to optimize.')
@click.option('--goal', prompt='goal',
            help='Specify the goal to optimize.')
@click.option('--trials_in_sweep', prompt='trials_in_sweep',
            help='Specify the number of trials in the sweep.')
@click.option('--accelerator', prompt='accelerator',
            help='Specify the accelerator to use.')
def run_wandb_sweep(run_dir, ae_save_dir, ae_idx, user_name, project_name, sweep_type, metric, goal, trials_in_sweep, accelerator):
    """
    Runs the hyperparameter tuning process for a neural network model by calling the wandb agent.

    Args:
        run_dir (str): The run directory to stor the ae models
        ae_save_dir (str): The directory to save the neural network model.
        ae_idx (str): The neural network model to tune.
        user_name (str): The username for the project.
        project_name (str): The name of the project.
        sweep_type (str): The type of sweep configuration.
        metric (str): The metric to optimize during tuning.
        goal (str): The goal of the metric (e.g., 'minimize' or 'maximize').
        trials_in_sweep (str): The number of trials to run in each sweep.
        accelerator (str): The accelerator to use for training.

    Returns:None
    """
    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_ENTITY"] = user_name
    os.environ['WANDB_DIR'] = 'runs'
    os.environ['WANDB_CONFIG_DIR'] = 'runs/wandb_config'
    os.environ['WANDB_CACHE_DIR'] = 'runs/wandb_cache'

    # Load the train and validation datsets for the fold and create dataloaders
    nn_params_dict = list_of_nn_params_dict[int(ae_idx)]
    nn_train_params_dict = list_of_nn_train_params_dict[int(ae_idx)]
    nn_datasets_dict = list_of_nn_datasets_dict[int(ae_idx)]
    train_dataset = load(f'runs/{run_dir}/{ae_save_dir}/datasets/train_dataset.pt', weights_only=False)
    if check_dict_key_exists('shuffle_data_between_epochs', nn_train_params_dict): # If exists then take value, otherwise default value
        shuffle = nn_train_params_dict['shuffle_data_between_epochs']
    else:
        shuffle = False
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=nn_train_params_dict['batch_size'],
                                    shuffle=shuffle,
                                    num_workers=0)
    val_dataset = load(f'runs/{run_dir}/{ae_save_dir}/datasets/val_dataset.pt', weights_only=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=nn_train_params_dict['batch_size'],
                                shuffle=False,
                                num_workers=0)
    # Check if any submodules require parameter optimization
    modules_to_optimize = []
    for module in list(nn_params_dict['modules'].keys()):
        if check_dict_key_exists('param_optimization', nn_params_dict['modules'][module]) and \
            nn_params_dict['modules'][module]['param_optimization']:
            modules_to_optimize.append(module)
    if len(modules_to_optimize) > 0:
        # Create a directory to store the hyperparameter runs
        ae_save_dir_path = f'runs/{run_dir}/{ae_save_dir}'
        params_to_tune = {}
        # Check for dictionary with 'values' key
        for module in modules_to_optimize:
            for module_key in list(nn_params_dict['modules'][module].keys()):
                if isinstance(nn_params_dict['modules'][module][module_key], dict) and \
                    check_dict_key_exists('values', nn_params_dict['modules'][module][module_key]):
                        params_to_tune[f'{module}-{module_key}'] = nn_params_dict['modules'][module][module_key]
        # Define the sweep configuration
        sweep_config = {
            'name': list_of_nn_params_dict[int(ae_idx)]['name'],
            'method': sweep_type,
            'metric': {
                'name': metric,
                'goal': goal
            },
            'parameters': params_to_tune
        }
        sweep_id = wandb.sweep(sweep_config)
        if int(trials_in_sweep) == -1:
            count = None
        else:
            count = int(trials_in_sweep)
        wandb.agent(sweep_id, lambda: run_wandb_agent(ae_save_dir_path, nn_params_dict, nn_train_params_dict, nn_datasets_dict, train_dataloader, val_dataloader, accelerator), count=count)
    else: # Dont optimize any submodule parameters and just do a single run.
        wandb.finish()
        run = wandb.init(job_type='training', resume=False, reinit=False, dir='runs')
        # The directory to store each individual run
        ae_train_runs_dir = f'runs/{run_dir}/{ae_save_dir}/ae_train_runs/{run.name}'
        # Create a run directory under tune_nn_params
        if not os.path.exists(ae_train_runs_dir):
            os.makedirs(ae_train_runs_dir)
        # Send all print statements to file for debugging
        print_file_path = f'{ae_train_runs_dir}/train_out.txt'
        sys.stdout = open(print_file_path, "w", encoding='utf-8')
        # Build the nn model
        ae = AE(ae_train_runs_dir, 
                nn_params_dict, 
                nn_train_params_dict, 
                f'runs/{run_dir}/{ae_save_dir}/datasets/combined_train_datasets.pt')
        ae.compile()
        print(' --> Model Compilation step complete.')
        if check_dict_key_exists('callbacks', nn_train_params_dict):
            callbacks = create_callback_object(nn_train_params_dict, ae_train_runs_dir)
        else:
            callbacks = None
        trainer = Trainer(max_epochs=nn_train_params_dict['epochs'],
                        accelerator=accelerator,
                        deterministic=True,
                        logger=WandbLogger(),
                        callbacks=callbacks,
                        enable_model_summary=False,
                        enable_progress_bar=False)
        trainer.fit(model=ae, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=None)
        wandb.finish()
        time.sleep(5)
        # Store the input dictionaries to the run directory
        pickle.dump(nn_params_dict, open(f'{ae_train_runs_dir}/nn_params_dict.pkl', 'wb'))
        pickle.dump(nn_train_params_dict, open(f'{ae_train_runs_dir}/nn_train_params_dict.pkl', 'wb'))
        pickle.dump(nn_datasets_dict, open(f'{ae_train_runs_dir}/nn_datasets_dict.pkl', 'wb'))

if __name__ == '__main__':
    run_wandb_sweep()
