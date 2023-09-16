" Plotting utilities "

import logging, os
import sys
logging.disable(logging.WARNING)
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import sys
from sklearn.decomposition import PCA
import pickle

sys.path.append('../')
print(os.getcwd())
# User defined libraries
#from utils.nn_utils import read_params_dict, load_trained_weights
from utils.custom_utils import create_ae_module
os.chdir('../')

def save_to_pickle(file, file_name, save_dir):
    
    with open(save_dir + '/' + file_name, 'wb') as f:
        pickle.dump(file, f)
        
    return None

def read_from_pickle(file_name, save_dir):
    
    with open(save_dir + '/' + file_name, 'rb') as f:
        file = pickle.load(f)
    
    return file

def load_csv_file(file, skip_header):
    # Load data from a text file, with missing values handled as specified.
    return np.genfromtxt(file, comments='#', delimiter=',', skip_header=skip_header)

def print_output_log(figure_dir, output_file_name):
    
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)

    sys.stdout = open(figure_dir + '/' + output_file_name, "w")
    
    print(f' --> User provided command line util argument : {util_type}')
    print(f' --> User provided command line dir argument : {run_dir}')
    print(f' --> User provided command line nn argument : {nn_index}')

# Common matploltib plot function (TODO:nthota2)
def plot(figsize, title, x, y, marker, color, label, xlabel, ylabel, xticks, yticks, xticklabels, yticklabels, font_dict):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, marker=marker, color=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_metrics():
    
    figure_dir = nn_save_dir + '/plots/metrics'
    print_output_log(figure_dir, 'plot_metrics_log.txt')

    with open(nn_save_dir + '/logs/csv_logs/' + 'val_logs.csv') as file:
        val_metric_names = file.readline().split(',')[1:]
        val_epoch_results = load_csv_file(file, skip_header=0)

    with open(nn_save_dir + '/logs/csv_logs/' + 'train_logs.csv') as file:
        train_metric_names = file.readline().split(',')[1:]
        train_epoch_results = load_csv_file(file, skip_header=0)

    # Get metric names to store as plot file name
    metric_names = []
    for val_metric in val_metric_names:
        metric_names.append(val_metric.partition('val_')[2].strip())

    epochs = train_epoch_results[:,0]

    for i, metric_name in enumerate(metric_names):
        fig, ax = plt.subplots(figsize=(5,5))
        metrics_train = np.array(train_epoch_results[:, i+1])
        # Remove the result from sanity check run
        metrics_val = np.array(val_epoch_results[1:, i+1])
        ax.plot(epochs, metrics_train, marker=None, c='b', label='train')
        ax.plot(epochs, metrics_val, marker=None, c='r', label='val')
        ax.plot(epochs[-1], metrics_train[-1], '.k', label=f'Metric at epoch {epochs[-1]} : {np.round(metrics_train[-1], 6)}')
        ax.plot(epochs[-1], metrics_val[-1], '.k', label=f'Val Metric at epoch {epochs[-1]} : {np.round(metrics_val[-1], 6)}')
        ax.set_xlabel('epochs')
        ax.legend()
        if metric_name == 'loss':
            ax.set_ylabel('total_loss')
            plt.savefig(nn_save_dir + f'/plots/metrics/total_loss.png')
            plt.clf()
            plt.close()
        else:
            ax.set_ylabel(metric_name)
            plt.savefig(nn_save_dir + f'/plots/metrics/{metric_name}.png')
            plt.clf()
            plt.close()
 
    print(' --> Saved metric plots to model directory')
    print(' --> Exit.')

def plot_pca_exp_var():

    ##### USER INPUT GOES HERE #####

    # Whether to load the latents or get them by running the model from last checkpoint
    load_latents = str(input(f' --> Load Latents (T/F) : '))
    latents_from_submodule = str(input(f' --> Latents from which submodule ? : '))

    ################################
    
    figure_file_path = nn_save_dir + '/plots/pca_exp_var/'
    print_output_log(figure_file_path, 'plot_pca_exp_var_log.txt')

    if load_latents == 'T' or load_latents == 'True':

        filepath = nn_save_dir + '/submodule_outputs/' + latents_from_submodule + '_output.csv'
        skip_header = 0
        latents = load_csv_file(filepath, skip_header)

    else:

        # Create a blank ae
        ae = create_ae_module(nn_save_dir,
                              nn_params_dict,
                              nn_train_params_dict,
                              nn_datasets_dict)
    
        # Load the latest ae from checkooint
        chpt_path = nn_save_dir + '/checkpoints/last.ckpt'
        loaded_ae = ae.load_from_checkpoint(chpt_path)

        # Name of the predictor attached to latent
        latents = loaded_ae(loaded_ae.all_samples)[latents_from_submodule].detach().numpy()
    
    print(' --> Successfully read nn latents!')
    print(f' --> Latents shape {latents.shape}') 
    
    pca = PCA(n_components=latents.shape[1])
    pca.fit(latents)
    
    num_features = np.arange(1, latents.shape[1] + 1, 1)
    yticks = np.arange(1, 100, 2)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlabel('Number of features')
    ax.set_ylabel('Explained Varianace')
    ax.plot(num_features, np.cumsum(np.round(pca.explained_variance_ratio_,3)*100),'.k')
    ax.set_yticks(yticks)
    ax.set_xticks(num_features)
    plt.savefig(nn_save_dir + '/plots/pca_exp_var/pca_exp_var.png')
    plt.clf()
    print(' --> Saved explained variance pca plot to model directory.')
    print(' --> Exit')

def plot_pca_2D():
    
    #params_dict_list = read_params_dict(run_dir, nn_index)
    
    #nn_save_dir = params_dict_list[0]
    #nn_params_dict = params_dict_list[1]
    #nn_dataset_dict = params_dict_list[4]
    
    figure_file_path = nn_save_dir + '/plots/pca_2D'
    print_output_log(figure_file_path, 'plot_pca_2D_log.txt')

    # Create a blank ae
    ae = create_ae_module(nn_save_dir,
                          nn_params_dict,
                          nn_train_params_dict,
                          nn_datasets_dict)

    # Load the latest ae from checkooint
    chpt_path = nn_save_dir + '/checkpoints/last.ckpt'
    loaded_ae = ae.load_from_checkpoint(chpt_path)

    # Name of the predictor attached to latent
    latents = loaded_ae(loaded_ae.all_inputs)['latent'].detach().numpy()
    
    # Plot for 3D PCA is only available to view through jupyter notebook
    pca_2D = PCA(n_components=2)
    latents_transformed_to_2D = pca_2D.fit_transform(latents)
    
    save_to_pickle(latents_transformed_to_2D, 'PCA_2D_data.pkl', figure_file_path)
    print(' --> Transformed points to 2D')
    print(' --> PCA 2D Stats : \n')
    # Principle axes in feature space representing direction of max variance
    print(f' --> Components : {pca_2D.components_} \n')
    print(f' --> Explained variance : {pca_2D.explained_variance_} \n')
    print(f' --> Explained variance ratio : {pca_2D.explained_variance_ratio_} \n')
    print(f' --> Per feature empirical mean : {pca_2D.mean_} \n')

    fig, ax = plt.subplots(figsize=(8,8))
    '''
    if 'y_layers' in nn_params_dict.keys():
        # We only use the first label to color code points in the latent
        y = np.squeeze(read_from_pickle('nn_y_pred.pkl', nn_save_dir)[0].numpy())
        
        # Check if any preprocessing done on the labels
        if nn_dataset_dict['y_preprocess_scheme']:
            y_preprocessors = params_dict_list[8]
            y_encoder_list = y_preprocessors[0]
            y_scaler_list = y_preprocessors[1]
            
            y_descr = params_dict_list[7]
            y_dtype_list = y_descr[0] 
            y_dim_list = y_descr[1]
            
            if y_dtype_list[0] == 'cat':
                c = y_encoder_list[0].inverse_transform(y.reshape(-1,1))
            else:
                c = y_scaler_list[0].inverse_transform(y.reshape(-1,1))
        else:
            c = y
                                                       
        cax = fig.add_axes([0.9, 0.15, 0.005, 0.7])
    else:
        c = 'k'
        cax = None
    '''
    c = 'k'
    cax=None
    
    pca_2D_data = ax.scatter(x=latents_transformed_to_2D[:,0], 
                             y=latents_transformed_to_2D[:,1], 
                             c=c,
                             marker='.', 
                             s=75, 
                             edgecolors='k')
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    
    if cax:
        fig.colorbar(pca_2D_data, cax=cax, orientation='vertical')
        
    plt.savefig(figure_file_path + '/pca_2D.png')
    plt.clf()
    
    print(' --> Saved 2D pca plot to model directory')
    print(' --> Exit.')

def plot_pearson_corr():

    ##### USER INPUT GOES HERE #####

    # Whether to load the latents or get them by running the model from last checkpoint
    load_latents = str(input(f' --> Load Latents (T/F) : '))
    latents_from_submodule = str(input(f' --> Latents from which submodule ? : '))

    ################################
    
    figure_file_path = nn_save_dir + '/plots/pearson_corr/'
    print_output_log(figure_file_path, 'plot_pearson_corr.txt')

    if load_latents == 'T':

        filepath = nn_save_dir + '/submodule_outputs/' + latents_from_submodule + '_output.csv'
        skip_header = 0
        latents = load_csv_file(filepath, skip_header)

    else:

        # Create a blank ae
        ae = create_ae_module(nn_save_dir,
                              nn_params_dict,
                              nn_train_params_dict,
                              nn_datasets_dict)
    
        # Load the latest ae from checkooint
        chpt_path = nn_save_dir + '/checkpoints/last.ckpt'
        loaded_ae = ae.load_from_checkpoint(chpt_path)

        # Name of the predictor attached to latent
        latents = loaded_ae(loaded_ae.all_samples)[latents_from_submodule].detach().numpy()

    print(' --> Successfully read nn latents!')
    print(f' --> Latents shape {latents.shape}') 
    
    num_samples = latents.shape[0]
    num_features = latents.shape[1]
    
    corr_matrix = np.empty(shape=(num_features, num_features), dtype=np.float32)
    
    for i in range(num_features):
        feature1 = latents[:,i]
        for j in range(num_features):
            feature2 = latents[:,j]
            
            f1 = feature1 - np.mean(feature1)
            f1_std = np.std(f1)
            f2 = feature2 - np.mean(feature2)
            f2_std = np.std(f2)

            covar = np.dot(f1, f2)/(num_samples - 1)
            
            corr = covar/(f1_std*f2_std)
            
            corr_matrix[i][j] = corr
            
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='bwr').get_figure()
    heatmap.savefig(figure_file_path + 'pearson_corr.jpeg')
    print(' --> Saved pearson correlation plot to model directory.')
    print(' --> Exit')
    
def plot_pred_error():
    
    figure_file_path = nn_save_dir + '/plots/pred_error/'
    print_output_log(figure_file_path, 'plot_pred_error_out.txt')
    
    y_pred = read_from_pickle('nn_y_pred.pkl', nn_save_dir)
    print(' --> Successfully read nn y predictions!')
    print(f' --> y prediction shape {len(y_pred)}')
    
    y_true = read_from_pickle('nn_y.pkl', nn_save_dir)
    print(' --> Successfully read nn y!')
    print(f' --> y true shape {len(y_true)}')
    
    for i in range(len(y_true)):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.scatter(y_true[i], y_pred[i].numpy(), marker='.', c='k')
        dotted_line = np.arange(np.min(y_true[i]), np.max(y_true[i]))
        ax.plot(dotted_line, dotted_line, linestyle='--', c='r')
        ax.set_xlabel('y true')
        ax.set_ylabel('y pred')
        ax.set_xlim(np.min(y_true[i]))
        ax.set_ylim(np.min(y_true[i]))
        plt.savefig(nn_save_dir + f'/plots/pred_error/pred_error_{i}.png')
        plt.clf()
    
    print(' --> Saved prediction error plot to model directory')
    print(' --> Exit.')

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Command line arguments for running plot_utils.py.')

    parser.add_argument('-u', '--util',
                        action='store',
                        required=True,
                        help='Specify which plotting utility you want to use. Supported types are')

    parser.add_argument('-d', '--dir', 
                        action='store',
                        required=True,
                        help='Specify the parent dir in which store all data')

    parser.add_argument('-n', '--nn', 
                        action='store', 
                        required=True,
                        help='Specify the nn for which train and test sets are being created')

    args = parser.parse_args()

    global util_type
    global nn_index 
    global run_dir 
    global nn_save_dir
    global nn_params_dict
    global nn_train_params_dict
    global nn_datasets_dict
                
    util_type = args.util
    nn_index = int(args.nn) - 1
    run_dir = '../runs/' + args.dir

    # Extract all required input data to run model from pickle files
    list_of_nn_params_dict = read_from_pickle('list_of_nn_params_dict.pkl', run_dir)
    list_of_nn_train_params_dict = read_from_pickle('list_of_nn_train_params_dict.pkl', run_dir)
    list_of_nn_datasets_dict = read_from_pickle('list_of_nn_datasets_dict.pkl', run_dir)
  
    nn_params_dict = list_of_nn_params_dict[nn_index]
    nn_train_params_dict = list_of_nn_train_params_dict[nn_index]
    nn_datasets_dict = list_of_nn_datasets_dict[nn_index]
                
    nn_save_dir = run_dir + '/' + nn_params_dict['model_type']
    
    plots_dir = nn_save_dir + '/plots'
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    
    # Common plt params for all plots
    #plt.rcParams.update({
    #    "text.usetex":True,
    #    "font.family":"serif",
    #    "font.serif":["Computer Modern Roman"]})
    
    if util_type == 'metrics':
        plot_metrics()
    elif util_type == 'pca_2D':
        plot_pca_2D()
    elif util_type == 'pca_exp_var':
        plot_pca_exp_var()    
    elif util_type == 'pred_error':
        plot_pred_error()
    elif util_type == 'pearson_corr':
        plot_pearson_corr()
    else:
        print(' --> Supported plotting utility not available')
