" Predict script "

import logging, os
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
import torch 
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning import Trainer
import sys
import click
import numpy as np

from nn.vanilla_ae import VanillaAE
from utils.custom_utils import set_global_random_seed, read_from_pickle

@click.command()
@click.option('--run_dir', prompt='run_dir', help='Specify the run dir where the model is located.')
@click.option('--ae', prompt='ae', help='Specify AutoEncoder number used for making the prediction.')
@click.option('--accelerator', prompt='accelerator', help='Specify the type of acceleration to use.')
@click.option('--submodule', prompt='submodule', help='Enter a submodule.')

def predict(run_dir, ae, accelerator, submodule):
    """ Predict script"""
    
    ae_index = int(ae) - 1

    run_dir = '../runs/' + run_dir

    list_of_nn_params_dict = read_from_pickle('list_of_nn_params_dict.pkl', run_dir)
    list_of_nn_train_params_dict = read_from_pickle('list_of_nn_train_params_dict.pkl', run_dir)
    list_of_nn_datasets_dict = read_from_pickle('list_of_nn_datasets_dict.pkl', run_dir)

    nn_params_dict = list_of_nn_params_dict[ae_index]
    nn_train_params_dict = list_of_nn_train_params_dict[ae_index]
    nn_datasets_dict = list_of_nn_datasets_dict[ae_index]

    global_seed = nn_train_params_dict['global_seed']

    nn_save_dir = run_dir + '/' + nn_params_dict['model_type']

    # Send all print statements to file for debugging
    print_file_path = nn_save_dir + '/' + 'predict_out.txt'
    sys.stdout = open(print_file_path, "w")

    print(f' --> User provided command line run_dir argument : {run_dir}')
    print(f' --> User provided command line ae argument : {ae}')
    print(f' --> User provided command line accelerator argument : {accelerator}')
    print(f' --> User provided command line submodule argument : {submodule}')

    set_global_random_seed(global_seed)

    print(f' --> Set global random seed {global_seed}.')

    # Pytorch accelerator information
    print(f' --> Number of threads : {torch.get_num_threads()}')
    print(f' --> Number of interop threads : {torch.get_num_interop_threads()}')

    # Create a blank ae

    new_ae = VanillaAE(nn_save_dir,
                       nn_params_dict,
                       nn_train_params_dict,
                       nn_datasets_dict)

    # Load the weights from the latest checkpoint
    chpt_path = nn_save_dir + '/checkpoints/last.ckpt'
    loaded_ae = new_ae.load_from_checkpoint(chpt_path)

    # Call the model on predict  dataset
    predict_dataset_dir = nn_save_dir + '/datasets/predict_dataset.pt'
    predict_dataset = torch.load(predict_dataset_dir)

    submodule_outputs_from_loaded = loaded_ae(predict_dataset)

    submodule_outputs_dir = nn_save_dir + '/submodule_outputs'
    if not os.path.exists(submodule_outputs_dir + '/predict'):
        os.mkdir(submodule_outputs_dir + '/predict')

    submodule_outputs = submodule_outputs_from_loaded[submodule].detach().numpy()
    
    filename = submodule + '_output.csv'
    np.savetxt(submodule_outputs_dir + '/predict/' + filename, 
               submodule_outputs, 
               delimiter=',')

if __name__ == '__main__':
    predict()
