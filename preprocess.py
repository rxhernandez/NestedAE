""" preprocess datasets script """

import os
import random

import click
from numpy import random as np_random # type: ignore
from numpy import round as np_round # type: ignore
from numpy import hstack, concatenate, split, savetxt # type: ignore
from pytorch_lightning import seed_everything # type: ignore
from torch import float32 as torch_float32 # type: ignore
from torch import manual_seed, tensor, save
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from NestedAE.dataset_utils import preprocess_datasets, TensorDataset

# User input dictionaries
from inputs.dataset_inputs import list_of_nn_datasets_dict
from inputs.train_inputs import list_of_nn_train_params_dict

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
    np_random.seed(seed)

@click.command()
@click.option('--run_dir', prompt='run_dir', help='Specify the run dir where the model is located.')
@click.option('--ae_save_dir', prompt='ae_save_dir', help='Specify the save directoryr for the autoencoder.')
@click.option('--ae_idx', prompt='ae_idx', help='Specify which autoencoder to train.')
@click.option('--dataset_type', prompt='dataset_type', help='Specify which dataset_type to work with.')
def run_preprocess(run_dir, ae_save_dir, ae_idx, dataset_type):
    """ Function to check input dictionaries and preprocess datasets.
    
    """
     # Check for runs directory
    if not os.path.exists('runs'):
        os.mkdir('runs')
        print(' --> Created runs directory.')
    else:
        print(' --> runs directory exists.')
    # Check for run_dir directory
    if not os.path.exists(f'runs/{run_dir}'):
        os.mkdir(f'runs/{run_dir}')
        print(f' --> runs/{run_dir} directory created.')
    else:
        print(f' --> runs/{run_dir} directory exists.')    
    # Check for ae_save_dir directory
    ae_save_dir_path = f'runs/{run_dir}/{ae_save_dir}'
    if not os.path.exists(ae_save_dir_path):
        os.mkdir(ae_save_dir_path)
        print(f' --> runs/{run_dir}/{ae_save_dir} directory created.')
    else:
        print(f' --> runs/{run_dir}/{ae_save_dir} directory exists.')
    # Create for wandb_cache directory
    if not os.path.exists('runs/wandb_cache'):
        os.makedirs('runs/wandb_cache')
        print(f' --> Created runs/wandb_cache directory.')
    else:
        print(f' --> runs/wandb_cache directory exists.')
    # Create for wandb_config directory
    if not os.path.exists('runs/wandb_config'):
        os.makedirs('runs/wandb_config')
        print(' --> Created runs/wandb_config directory.')
    else:
        print(' --> runs/wandb_config directory exists.')

    # Load the input python dictionaries
    ae_idx = int(ae_idx)
    nn_datasets_dict = list_of_nn_datasets_dict[ae_idx]
    nn_train_params_dict = list_of_nn_train_params_dict[ae_idx]

    # Set a global random seed
    global_seed = nn_train_params_dict['global_seed']

    dataset_save_dir_path = ae_save_dir_path + '/datasets'
    if not os.path.exists(dataset_save_dir_path):
        os.mkdir(dataset_save_dir_path)
        print(f' --> {dataset_save_dir_path} directory created.')
    else:
        print(f' --> {dataset_save_dir_path} directory already exists. Proceeding to rewrite.')

    # Preprocess the dataset and get the sklearn preprocessing objects.
    processed_dataset, desc_preprocessors = preprocess_datasets(dataset_save_dir_path, nn_datasets_dict, dataset_type) 
    desc_names = list(processed_dataset.keys())
    
    ############################################################################
    # Choose how to split the dataset into train and test sets
    ############################################################################

    val_split = nn_train_params_dict['val_split']

    # Method 1 : Split the dataset into 'k'-folds
    # create_kfold_datasets(dataset, desc_preprocessors, dataset_save_dir_path, 5, global_seed)

    # # Method 2 : Random splitting
    # idxs = list(range(0, len(dataset)))
    # random.shuffle(idxs)
    # train_idxs = idxs[:int(len(dataset) * (1 - val_split))]
    # val_idxs = idxs[int(len(dataset) * (1 - val_split)):]
    # dictionary_to_TorchDataset(dataset, desc_preprocessors, 
    #                            dataset_save_dir_path=dataset_save_dir_path,
    #                            dataset_name=f'train', 
    #                            idxs=train_idxs)    
    # dictionary_to_TorchDataset(dataset, desc_preprocessors,
    #                             dataset_save_dir_path=dataset_save_dir_path,
    #                             dataset_name=f'val', 
    #                             idxs=val_idxs)
    
    # Method 3 : Spectral clustering before randomly spliting each cluster into train, test set
    ##### User input fields #####
    numpy_dataset = hstack([processed_dataset[desc_name] for desc_name in desc_names])
    run_optimal_cluster_search = False
    num_cluster_range = range(2, 10) # Active if above set to True
    num_clusters = 3
    ##############################
    print(f' --> Starting clustering then random splitting strategy.')
    if run_optimal_cluster_search:
        calinski_harabasz_scores = [] # (sum between cluster disperion)/(sum within cluster dispersion). Higher the score, better the model clustering 
        davies_bouldin_index = [] # Measures average "similarity" between clusters. 0 is lowest. Closer to 0, the better.
        silhouette_scores = [] # Defined for each sample as (b - a)/max(a,b). a=mean dist. bw. sample and other samples in class. b=mean dist. bw. sample and other samples in next nearest cluster. Higher the better.
        for num_cluster in num_cluster_range:
            spectral_cluster = SpectralClustering(n_clusters=num_cluster, random_state=global_seed)
            spectral_cluster.fit(numpy_dataset)
            cluster_labels = spectral_cluster.labels_
            calinski_harabasz_scores.append(calinski_harabasz_score(numpy_dataset, cluster_labels))
            davies_bouldin_index.append(davies_bouldin_score(numpy_dataset, cluster_labels))
            silhouette_scores.append(silhouette_score(numpy_dataset, cluster_labels))
        # Plotting cluster metrics 
        _, ax = plt.subplots(3, 1, figsize=(20, 6), sharex=True)
        ax[0].plot(num_cluster_range, calinski_harabasz_scores)
        ax[0].set_ylabel('Calinski-Harabasz Index')
        ax[0].grid()
        ax[1].plot(num_cluster_range, davies_bouldin_index, label='Davies-Bouldin Index')
        ax[1].set_ylabel('Davies-Bouldin Index')
        ax[1].grid()
        ax[2].plot(num_cluster_range, silhouette_scores, label='Silhouette Score')
        ax[2].set_ylabel('Silhouette Score')
        ax[2].set_xlabel('Number of Clusters')
        ax[2].set_xticks(num_cluster_range)
        ax[2].set_xlim([num_cluster_range[0], num_cluster_range[-1]])
        ax[2].grid()
        # Save figure to disk
        plt.tight_layout()
        plt.savefig(dataset_save_dir_path + '/cluster_metrics.pdf')
    # Selcting best performing number of clusters
    spectral_cluster = SpectralClustering(n_clusters=num_clusters, random_state=global_seed)
    spectral_cluster.fit(numpy_dataset)
    # Split datapoints into train and validation sets based on their labels
    train_samples_list = []
    val_samples_list = []
    for i in range(num_clusters):
        cluster_samples = numpy_dataset[spectral_cluster.labels_ == i]
        if len(cluster_samples) == 1:
            train_samples_list.append(cluster_samples)
            continue
        cluster_train, cluster_val = train_test_split(cluster_samples, test_size=val_split, random_state=global_seed)
        train_samples_list.append(cluster_train)
        val_samples_list.append(cluster_val)
    # Row stacking all samples
    train_samples_vstacked = concatenate(train_samples_list)
    val_samples_vstacked = concatenate(val_samples_list)
    savetxt(dataset_save_dir_path + '/train_samples_vstacked.csv', train_samples_vstacked, 
            delimiter=',', header=','.join(desc_names))
    savetxt(dataset_save_dir_path + '/val_samples_vstacked.csv', val_samples_vstacked,
            delimiter=',', header=','.join(desc_names))
    print(f' --> Actual val split ratio : {len(val_samples_vstacked) / (len(train_samples_vstacked) + len(val_samples_vstacked))}')
    # Plot the descriptor distributions
    _, ax = plt.subplots(5, 5, figsize=(20, 20))
    # Plot the feature histograms 
    for i in range(train_samples_vstacked.shape[1]):
        ax[i//5, i%5].hist(train_samples_vstacked[:, i], bins=10, label='train', alpha=0.5)
        ax[i//5, i%5].hist(val_samples_vstacked[:, i], bins=10, label='val', alpha=1)
        ax[i//5, i%5].set_title(f' ks test : {np_round(ks_2samp(train_samples_vstacked[:, i], val_samples_vstacked[:, i]).pvalue, 4)}')
    plt.tight_layout()
    plt.savefig(dataset_save_dir_path + '/desc_dist.pdf')
    # Covert numpy_dataset to tensor_dictionary
    train_tensor_dictionary = {}
    val_tensor_dictionary = {}
    desc_shapes = [value.shape[1] for value in processed_dataset.values()]
    splitting_idxs = [0] * len(desc_shapes)
    # Cumulative sum of the desc_shapes
    for i in range(len(desc_shapes)):
        if i == 0:
            splitting_idxs[i] = desc_shapes[i]
        else:
            splitting_idxs[i] = desc_shapes[i] + splitting_idxs[i - 1]
    print(f' --> Splitting combined numpy dataset at indices : {splitting_idxs}')
    # Split the numpy dataset at the splitting idxs
    train_samples_col_split = split(train_samples_vstacked, splitting_idxs, axis=1)
    val_samples_col_split = split(val_samples_vstacked, splitting_idxs, axis=1)
    for i, desc_name in enumerate(desc_names):
        train_tensor_dictionary[desc_name] = tensor(train_samples_col_split[i], dtype=torch_float32)
        val_tensor_dictionary[desc_name] = tensor(val_samples_col_split[i], dtype=torch_float32)
    # Convert tensor_dictionary to TensorDataset
    train_tensor_dataset = TensorDataset(name='train',
                                         tensor_dictionary=train_tensor_dictionary,
                                         desc_preprocessors=desc_preprocessors)
    save(train_tensor_dataset, dataset_save_dir_path + f'/train_dataset.pt')
    val_tensor_dataset = TensorDataset(name='val',
                                       tensor_dictionary=val_tensor_dictionary,
                                       desc_preprocessors=desc_preprocessors)
    save(val_tensor_dataset, dataset_save_dir_path + f'/val_dataset.pt')
    
if __name__ == '__main__':
    run_preprocess()
