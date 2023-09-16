" Plotting utilities "

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn.decomposition import PCA
from custom_utils import read_from_pickle, save_to_pickle

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

def plot_metrics(nn_save_dir):
    
    figure_dir = nn_save_dir + '/plots/metrics'
    print_output_log(figure_dir, 'plot_metrics_out.txt')
    
    #if nn_compile_params_dict['y_metrics'] == None:
    #    num_metrics = len(nn_compile_params_dict['X_metrics']) + 1
    #else:
    #    num_metrics = len(nn_compile_params_dict['X_metrics'] + nn_compile_params_dict['y_metrics']) + 1

    with open(nn_save_dir + '/' + 'epoch_results.csv') as file:
        names = file.readline().split(',')
        epoch_results = np.genfromtxt(file, comments='#', delimiter=',', skip_header=0)
    
    metric_names = names[1:]
    epochs = np.array(epoch_results[0:,0]) + 1
    #for i in range(1, num_metrics + 1)
    for i in range(0, int(len(metric_names)/2)):
        fig, ax = plt.subplots(figsize=(5,5))
        metrics_train = np.array(epoch_results[0:, i + 1])
        metrics_val = np.array(epoch_results[0:, i + 1 + int(len(metric_names)/2)])
        ax.plot(epochs, metrics_train, marker=None, c='b', label='train')
        ax.plot(epochs, metrics_val, marker=None, c='r', label='val')
        ax.set_xlabel('epochs')
        ax.set_ylabel(metric_names[i])
        #ax.set_xticks(np.arange(1, epochs[-1], 10), labels=None, minor=False)# Sets the major ticks
        ax.legend()
        plt.savefig(nn_save_dir + f'/plots/metrics/{metric_names[i]}.png')
        plt.clf()
    
    print(' --> Saved metric plots to model directory')
    print(' --> Exit.')
          
def plot_latents(nn_save_dir, supervised_latents):
    
    figure_file_path = nn_save_dir + '/plots/latents'
    print_output_log(figure_file_path , 'plot_latents_out.txt')
    
    latents = read_from_pickle('nn_latents.pkl', nn_save_dir)
    print(' --> Successfully read nn latents!')
    print(f' --> Latents shape {latents.shape}')

    # Load the latents from model save dir
    latents = read_latents()
    if supervised_latents:
        y_pred = read_from_pickle('nn_y_pred.pkl', nn_save_dir)
        print(' --> Successfully read nn y predictions!')
        print(f' --> y prediction shape {y_pred.shape}')
        
    if latents.shape[1] == 2:
        fig_2D, ax_2D = plt.subplots(figsize=(10,10))
        ax_2D.plot(latents[:,0], latents[:,1], '.k')
        if supervised_latents:
            ax_2D.plot(latents[:,0], latents[:,1], c=y_pred)
    elif latents.shape[2] == 1:
        pass
    else:
        print(' --> This function can only plot 1D and 2D latent spaces')

def plot_pca_2D(nn_save_dir, supervised_latents):
    
    figure_file_path = nn_save_dir + '/plots/pca_2D'
    print_output_log(figure_file_path, 'plot_pca_2D_out.txt')
    
    latents = read_from_pickle('nn_latents.pkl', nn_save_dir)
    print(' --> Successfully read nn latents!')
    print(f' --> Latents shape {latents.shape}') 
    
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
    if supervised_latents:
        c = read_from_pickle('nn_y_pred.pkl', nn_save_dir)
        cax = fig.add_axes([0.9, 0.15, 0.005, 0.7])
    else:
        cax = None
        c = 'k'
    pca_2D_data = ax.scatter(x=latents_transformed_to_2D[:,0], 
                            y=latents_transformed_to_2D[:,1], 
                            c=c, 
                            cmap='gist_rainbow', 
                            marker='.', 
                            s=25, 
                            edgecolors='k')
    fig.colorbar(pca_2D_data, cax=cax, orientation='vertical')
    plt.savefig(figure_file_path + '/pca_2D.png')
    plt.clf()
    print(' --> Saved 2D pca plot to model directory')
    print(' --> Exit.')
    
def plot_pca_exp_var(nn_save_dir):
    
    figure_file_path = nn_save_dir + '/plots/pca_exp_var/'
    print_output_log(figure_file_path, 'plot_pca_exp_var_out.txt')
    
    latents = read_from_pickle('nn_latents.pkl', nn_save_dir)
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
    
def plot_conf_matrix(nn_save_dir):
    pass
    
def plot_pred_error(nn_save_dir):
    
    figure_file_path = nn_save_dir + '/plots/pred_error/'
    print_output_log(figure_file_path, 'plot_pred_error_out.txt')
    
    y_pred = read_from_pickle('nn_y_pred.pkl', nn_save_dir)
    print(' --> Successfully read nn y predictions!')
    print(f' --> y prediction shape {len(y_pred)}')
    
    y_true = read_from_pickle('nn_y.pkl', nn_save_dir)
    print(' --> Successfully read nn y!')
    print(f' --> y true shape {len(y_true)}')
    
    dotted_line = np.arange(np.min(y_true), np.max(y_true))
    for i in eranega(y_pred):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot(y_true, y_pred, marker=None, c='k')
        ax.plot(dotted_line, dotted_line, marker=None, c='-r')
        ax.set_xlabel('y true')
        ax.set_ylabel('y pred')
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
                
    util_type = args.util
    nn_index = int(args.nn) - 1
    run_dir = '../../runs/' + args.dir

    # Extract all required input data to run model from pickle files
    list_of_nn_compile_params_dict = read_from_pickle('list_of_nn_compile_params_dict.pkl', run_dir)
    list_of_nn_params_dict = read_from_pickle('list_of_nn_params_dict.pkl', run_dir)
  
    nn_compile_params_dict = list_of_nn_compile_params_dict[nn_index]
    nn_params_dict = list_of_nn_params_dict[nn_index]
                
    nn_save_dir = run_dir + '/' + nn_params_dict['model_name']
    
    plots_dir = nn_save_dir + '/plots'
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    
    # Common plt params for all plots
    #plt.rcParams.update({
    #    "text.usetex":True,
    #    "font.family":"serif",
    #    "font.serif":["Computer Modern Roman"]})
    
    if util_type == 'metrics':
          plot_metrics(nn_save_dir)
    elif util_type == 'latents':
          plot_latents(nn_save_dir, nn_params_dict['add_supervision_on_latent'])
    elif util_type == 'pca_2D':
          plot_pca_2D(nn_save_dir, nn_params_dict['add_supervision_on_latent'])
    elif util_type == 'pca_exp_var':
          plot_pca_exp_var(nn_save_dir)
    elif util_type == 'conf_matrix':
          plot_conf_matrix(nn_save_dir)      
    elif util_type == 'pred_error':
          plot_pred_error(nn_save_dir)
    else:
        print(' --> Supported plotting utility not available')