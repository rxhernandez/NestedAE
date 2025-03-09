This user guide is intended to provide a brief overview of the NestedAE codebase and how to use it.

### Directory structure

The main directories in NestedAE are code and datasets.<br>
<pre>

1. code - contains the code for NestedAE.<br>
    |<br>
    |-> go.sh : This is the script that should be run for data preprocessing, model training, model inference.<br>
    |-> inputs_example : Contains templates for input files.<br>
    |-> inputs_perov_data : Contains input files for training NestedAE on the multiscale perovskite dataset.<br>
    |-> inputs_syn_data : Contains input files for training NestedAE on the synthetic dataset.<br>
    |-> nn : This directory contains the code for different model types.<br>
    |    |<br>
    |    |-> vanilla_ae.py : Contains the VanillaAE class.<br>
    |<br>
    |-> predict.py: Contains the code for model inference.<br>
    |-> preprocess.py: Calls create_preprocessed_datasets() from utils/dataset_utils.py<br>
    |-> train.py: Contains the code for model training.<br>
    |-> utils :<br>
         |<br>
         |-> plot_utils.py : Contains the plotting functions. For inline plotting capabilities,<br> 
         |                       add your custom plotting code to this file.<br>
         |-> dataset_utils.py : Contains data preprocessing functions that are called by preprocess.py<br>
         |-> nn_utils.py : Contains the model training and inference functions that are called by train.py and predict.py<br>
         |-> custom_utils.py : Contains custom functions for setting global random seed, pickle and unpickle files.<br>
         |-> analysis.ipynb : Contains code for replicating the plots in the paper.<br>

2. datasets - this is where the user should store their datasets.<br>
    |<br>
    |-> PSC_bandgaps<br>
    |        |<br>
    |        |-> PSC_bandgaps_dataset.csv : This file contains the bandgaps of 499 hybrid metal halide perovskites<br>
    |<br>
    |-> PSC_efficiencies<br>
    |        |<br>
    |        |-> PSC_efficiencies_dataset.csv : This file contains the efficiencies of 2018 hybrid metal halide perovskites<br>
    |        |-> jacobsson2021dataset.csv<br>
    |        |-> create_efficiencies_dataset.ipynb : Code to generate the PSC_efficiencies_dataset.csv file<br>
    |<br>        
    |-> synthetic_dataset<br>
             |<br>
             |-> synthetic_dataset.csv : This file contains the synthetic dataset used in the paper<br>
             |-> synthetic_data_FuncAddNoise_200.csv : This file contains the synthetic dataset with added noise<br>
             |-> SynthDataset.ipynb: Code to generate the synthetic_dataset.csv and synthetic_data_FuncAddNoise_200.csv<br>

</pre>
### How to use NestedAE

* Create a new conda environment for NestedAE by following the below commands
    - Create a conda environment with python=3.9.18 : conda create -n your_env_name python=3.9.18
    - Update your_env_name with pip packages specified in nestedae.yml : conda env update --file nestedae.yml
* Inside the 'code' directory, create a new inputs directory and populate with the required input files. A template of the input files is provided in "inputs_example". <span style="color: red">**Copy and paste the files into your input dir. but do not change the input file names**</span>
* Edit the user section in go.sh script. 
* Run the go.sh scrip (./go.sh) from the code directory. This will create the run directory (if not present) and populate it with model directory.
* The first program run by go.sh is the preprocess_data.py script. This script checks for missing dictionary keys in the input files, pickles the input dictionaries and stores them in the model directory, creates a run_summary.txt file and calls the <span style="color: orange;">**create_preprocessed_datasets()**</span>function from utils/dataset_utils.py that starts the data preprocessing step.
* In 'train' mode, <span style="color: orange;">**create_preprocessed_datasets()**</span> will split the input dataset into train and validation set by default. To enable 'k' splits of the dataset set the <span style="color: orange;">**create_preprocessed_datasets()**</span> flag to True in create_preprocessed_datasets() function. The 'k' splits will be stored in the datasets directory of the model directory. Choose the split you want to train the model with and rename it as train_dataset.pt and val_dataset.pt.
* Logs of preprocess_data.py are stored in the preprocess_data_out.txt in the model directory.
* The second program run by go.sh is train.py if mode is set to 'train'. Else if in 'predic' mode, the predict.py script is run. 
* Logs of train.py and predict.py are stored in the train_out.txt and predict_out.txt files respectively in the model directory.
* Hyper parameter optimization can be done by creating several model directories and running training for each model in parallel. This is possible as input files for each model are pickled and stored in the model directory.
* Once training is complete for the first autoencoder. Training for the next one can be started by simply specifying ae=2 and mode=train in the go.sh script. Ensure you are still in the same model directory.
* Repeat for remaining nested autoencoders.
* To plot metrics run the following command from the utils folder : python3 plot_utils.py -u metrics -d model_name_goes_here -n ae_to_plot_metrics_for. 
* To replicate the results in the paper follow the steps below :
    - Change the "inputs_dir" variable to "inputs_perov_data" to run NestedAE on the multiscale perovskite dataset or to "inputs_syn_data" to run NestedAE on the synthetic dataset in go.sh script.
    - To generate the plots in the paper head over to the analysis.ipynb file.
