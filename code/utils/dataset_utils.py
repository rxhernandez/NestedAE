" Dataset utils script "

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

from utils.nn_utils import check_dict_key_exists

class CreateDataset(Dataset):
    """ Creates a PyTorch Dataset. """

    def __init__(self, name, dataset, variable_names,
                 variable_shapes, variable_preprocessors, variable_dtypes):

        # List of tensors
        self.name = name
        self.dataset = dataset

        self.variable_names = variable_names
        self.variable_shapes = variable_shapes
        self.variable_preprocessors = variable_preprocessors
        self.variable_dtypes = variable_dtypes

        # Get all the variable shapes
        shapes = list(self.variable_shapes.values())

        num_samples = shapes[0][0]

        sum_of_variables_shape = sum([shape[1] for shape in shapes])

        self.shape = (num_samples, sum_of_variables_shape)

    # Returns the number of samples in each dataset
    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        item = {}
        for variable_name in self.variable_names:
            item[variable_name] = self.dataset[variable_name][idx, :]

        return item

def load_csv_file(file, header, skiprows):
    """ Reads the csv file into a pandas dataframe
    
    Args:
        header : Row (0-indexed) to use as column labels for the dataframe
        index_col : Column (0-indexed) to use as the row labels for the datafram
    
    Returns : numpy array
    """
    dataframe = pd.read_csv(file, header=header, skiprows=skiprows)
    return dataframe

def load_xlsx_file(file, sheet_name, header, skiprows):
    """ Reads the excel file into a pandas dataframe
    
    Args:
        sheet_name : Name of the excel sheet to read.
        header : Row (0-indexed) to use as column labels for the dataframe
        index_col : Column (0-indexed) to use as the row labels for the datafram
    
    Returns : numpy array
    """
    dataframe = pd.read_excel(file, sheet_name=sheet_name, 
                              header=header, skiprows=skiprows)
    return dataframe

def load_npy_file(file):
    """ Loads the .npy, .npz or pickled files into a numpy array"""
    return np.load(file)

def one_hot_encode(samples, variable):
    """ One hot encodes the samples for the variable"""
    _ohe = OneHotEncoder(dtype=np.float32)
    _ohe.fit(samples)
    samples_ohe = _ohe.transform(samples).toarray()

    categories = _ohe.categories_[0]

    dtype = samples_ohe.dtype
    shape = samples_ohe.shape

    print(f' --> {variable} dtype : {dtype}')
    print(f' --> {variable} dim   : {shape}')

    return samples_ohe, _ohe, dtype, shape, categories
   
def label_binarizer(samples, variable):
    """ Label binarizes the samples for the variable"""
    _lb = LabelBinarizer(dtype=np.float32)
    _lb.fit(samples)
    samples_lb = _lb.transform(samples)

    dtype = samples_lb.dtype
    shape = samples_lb.shape

    print(f' --> {variable} dtype : {dtype}')
    print(f' --> {variable} shape : {shape}')

    return samples_lb, _lb, dtype, shape

def label_encoder(samples, variable):
    """Label encodes the samples for the variable"""
    _le = LabelEncoder()
    _le.fit(samples)
    samples_le = _le.transform(samples)

    dtype = samples_le.dtype
    shape = samples_le.shape

    print(f' --> {variable} dtype : {dtype}')
    print(f' --> {variable} shape : {shape}')

    return samples_le, _le, dtype, shape

def standardize(samples, variable):
    """ Standardizes the samples for the variable"""
    samples = samples.astype(np.float32)
    _ss = StandardScaler()
    samples_std = _ss.fit_transform(samples)

    dtype = samples_std.dtype
    shape = samples_std.shape

    print(f' --> {variable} dtype : {dtype}')
    print(f' --> {variable} shape : {shape}')

    return samples_std, _ss, dtype, shape

def create_preprocessed_datasets(nn_save_dir, nn_dataset_dict, global_seed, test_split=0.2, mode='train'):
    """ Reads data from specified source and processes it according to specifications in nn_dataset_dict
    
    Args:
        nn_save_dir: Model directory (Required)
        nn_dataset_dict: The dictionary containing all 
        preprocessing specifications for the dataset (Required)
        global_seed: The seed used for shuffling (Required)
        
    Returns: None
    """
    datasets = nn_dataset_dict[mode]

    dataset_dicts = list(datasets.values())
    dataset_names = list(datasets.keys())

    # Create a directory to store all the datasets
    # Check if run directory already exists
    dataset_save_dir = nn_save_dir + '/datasets'
    if os.path.exists(dataset_save_dir) is False:
        os.mkdir(dataset_save_dir)
        print(' --> Dataset directory created.')
    else:
        print(' --> Dataset directory already exists. Proceeding to rewrite.')

    # Create one dataset that combines data from all datasets
    dataset = {}
    tensor_dataset = {}
    variable_preprocessors = {}
    variable_dtypes = {}
    variable_shapes = {}
    all_variable_names = []
    sample_wts = None

    # Outer loop iterates over each dataset (Ex : X, y, latents ...)
    for i, dataset_dict in enumerate(dataset_dicts):

        dataset_name = dataset_names[i]

        #nn_name = nn_save_dir.split('/')[-1]
    
        dataset_file_path = dataset_dict['path']
        dataset_file_name = dataset_file_path.split('/')[-1]
        dataset_file_type = dataset_file_name.split('.')[-1]

        if dataset_file_type == 'npy':
            print(f' --> Found .npy file : {dataset_file_name}')
            dataframe = load_npy_file(dataset_file_path)
        elif dataset_file_type == 'xlsx':
            print(f' --> Found .xlsx file : {dataset_file_name}')
            try:
                sheet_name = dataset_dict['sheet_name']
                header = dataset_dict['header']
                skiprows = dataset_dict['skiprows']
            except:
                sheet_name = 0
                header = 0
                skiprows = None
            dataframe = load_xlsx_file(dataset_file_path, sheet_name, 
                                       header, skiprows)
        elif dataset_file_type == 'csv':
            print(f' --> Found .csv file : {dataset_file_name}')
            try:
                header = dataset_dict['header']
                skiprows = dataset_dict['skiprows']
            except:
                header = 0
                skiprows = None
            dataframe = load_csv_file(dataset_file_path, 
                                      header, skiprows)
        else:
            raise FileNotFoundError(' --> Supported file type not found')
            
        print(f' --> Loaded {dataset_name} as a dataframe.')
        print(f' --> {dataset_name} Dataframe shape : {dataframe.shape}')

        print(' --> Dataframe head.')
        print(dataframe.head())

        # data_id is string identifier for the pandas df col / df cols
        variable_dicts = list(dataset_dict['variables'].values())
        variable_names = list(dataset_dict['variables'].keys())

        if i == 0:
            all_variable_names = variable_names
        else:
            all_variable_names.extend(variable_names)

        # Inner loop iterates over each variable in the dataset
        for j, variable_dict in enumerate(variable_dicts):

            variable_name = variable_names[j]

            cols = variable_dict['cols']
            cols = cols if not isinstance(cols, list) else list(cols)

            preprocess_scheme = variable_dict['preprocess']

            # Check for any NA values in variable
            na_values = dataframe.iloc[:, cols].isna()

            if na_values.any().any():
                raise ValueError(f' --> NA values found in {variable_name} \
                        dataframe. Check log file for details.')
            else:
                print(f' --> No NA values found in {variable_name}.')

            # Extract samples for variable
            samples = dataframe.iloc[:, cols].values

            if samples.shape[1] == 1:
                samples = samples.reshape(-1, 1)

            print(f' --> Extracting data for {variable_name} from {dataset_name} dataframe cols {cols}.')

            #
            # Data Preprocessing section
            #

            if preprocess_scheme is None:
                samples = samples.astype(np.float32)

                print(f' --> No preprocessing done for {variable_name} \
                            from {dataset_name} dataframe cols {cols}.')

                preprocessor = None
                dtype = samples.dtype
                shape = samples.shape

                print(f' --> {variable_name} dtype : {dtype}')
                print(f' --> {variable_name} dim   : {shape}')
            elif preprocess_scheme == 'std':
                samples, preprocessor, dtype, shape = standardize(samples, variable_name)
            elif preprocess_scheme == 'ohe':
                samples, preprocessor, dtype, shape, categories = one_hot_encode(samples, variable_name)
                print(f' --> Encoded col {cols} \
                      with {len(categories)} \
                      categories {categories}')
            elif preprocess_scheme == 'lb':
                samples, preprocessor, dtype, shape = label_binarizer(samples, variable_name)
            elif preprocess_scheme == 'le':
                samples, preprocessor, dtype, shape = label_encoder(samples, variable_name)
            else:
                raise ValueError(' --> Preprocessing scheme not defined. ')

            if i==0 and j==0:
                samples_preview = samples
            else:
                #print(samples_preview)
                #print(samples.shape)
                #print(samples)
                samples_preview = np.hstack((samples_preview, samples))

            dataset[variable_name] = samples
            tensor_dataset[variable_name] = torch.tensor(samples, dtype=torch.float32)
            
            variable_preprocessors[variable_name] = preprocessor
            variable_dtypes[variable_name] = dtype
            variable_shapes[variable_name] = shape
    
        if check_dict_key_exists('weight_samples', dataset_dict) is True:

            sample_wts = weight_samples(dataframe, dataset_dict)
            all_variable_names.append('sample_wts')

            # Add the sample wts to the preview
            samples_preview = np.hstack((samples_preview, sample_wts))

            dataset['sample_wts'] = sample_wts
            tensor_dataset['sample_wts'] = torch.tensor(sample_wts, dtype=torch.float32)
            
            variable_preprocessors['sample_wts'] = None
            variable_dtypes['sample_wts'] = sample_wts.dtype
            variable_shapes['sample_wts'] = sample_wts.shape

            print(f' --> sample_wts dtype : {sample_wts.dtype}')
            print(f' --> sample_wts shape : {sample_wts.shape}')

    if mode == 'train':

        # Save the processed data to .csv file for easy preview
        preview_file_name = 'dataset_preview.csv'
        np.savetxt(dataset_save_dir + '/' + preview_file_name, samples_preview, delimiter=',')

        use_kfold_cross_validator = True

        if use_kfold_cross_validator:
            ae_dataset = CreateDataset(name=dataset_name,
                                    dataset=tensor_dataset,
                                    variable_names=all_variable_names,
                                    variable_shapes=variable_shapes,
                                    variable_preprocessors=variable_preprocessors,
                                    variable_dtypes=variable_dtypes) 

            pickle_file_name = 'dataset.pt'
            torch.save(ae_dataset, dataset_save_dir + '/' + pickle_file_name)

            n_splits = 5
            cross_val_type = 'kfold'
            class_col = 'bg_class'

            train_tensor_datasets, \
            train_tensor_datasets_var_shapes, \
            val_tensor_datasets, \
            val_tensor_datasets_var_shapes = create_kfold_datasets(cross_val_type, dataset,
                                                                   n_splits, global_seed,
                                                                   class_col=class_col)

            # Preview each train k-fold dataset
            for i, train_tensor_dataset in enumerate(train_tensor_datasets):
                
                variable_names = list(train_tensor_dataset.keys())

                # Create the PyTorch train dataset
                ae_train_dataset = CreateDataset(name=dataset_name,
                                                dataset=train_tensor_dataset,
                                                variable_names=variable_names,
                                                variable_preprocessors=variable_preprocessors,
                                                variable_dtypes=variable_dtypes,
                                                variable_shapes=train_tensor_datasets_var_shapes[i]) 

                pickle_file_name = f'train_dataset_fold_{i}.pt'
                torch.save(ae_train_dataset, dataset_save_dir + '/' + pickle_file_name)

                ae_val_dataset = CreateDataset(name=dataset_name,
                                                dataset=val_tensor_datasets[i],
                                                variable_names=variable_names,
                                                variable_preprocessors=variable_preprocessors,
                                                variable_dtypes=variable_dtypes,
                                                variable_shapes=val_tensor_datasets_var_shapes[i]) 

                pickle_file_name = f'val_dataset_fold_{i}.pt'
                torch.save(ae_val_dataset, dataset_save_dir + '/' + pickle_file_name)

                j = 0
                for train_samples, val_samples in zip(train_tensor_dataset.values(), val_tensor_datasets[i].values()):
                    if j == 0:
                        train_samples_preview = train_samples
                        val_samples_preview = val_samples
                    else:
                        train_samples_preview = np.hstack((train_samples_preview, train_samples))
                        val_samples_preview = np.hstack((val_samples_preview, val_samples))
                    j += 1
                # Save the train and val samples preview to a csv
                np.savetxt(dataset_save_dir + '/' + f'train_dataset_preview_fold_{i}.csv', 
                            train_samples_preview, delimiter=',', header=','.join(variable_names))
                np.savetxt(dataset_save_dir + '/' + f'val_dataset_preview_fold_{i}.csv', 
                            val_samples_preview, delimiter=',', header=','.join(variable_names))
        else:
            train_tensor_dataset,\
            train_var_shapes, \
            val_tensor_dataset, \
            val_var_shapes = create_train_val_datasets(dataset, test_split, 
                                                       global_seed, dataset_save_dir)
            
            print(' --> Created train and val datasets for dataset.')
            print(f' --> Number of variables in dataset {len(dataset)}.')
            print(f' --> Train dataset shape : {train_var_shapes}.')
            print(f' --> Val dataset shape :{val_var_shapes}.')

            ae_dataset = CreateDataset(name=dataset_name,
                                    dataset=tensor_dataset,
                                    variable_names=all_variable_names,
                                    variable_shapes=variable_shapes,
                                    variable_preprocessors=variable_preprocessors,
                                    variable_dtypes=variable_dtypes) 

            pickle_file_name = 'dataset.pt'
            torch.save(ae_dataset, dataset_save_dir + '/' + pickle_file_name)

            ae_train_dataset = CreateDataset(name=dataset_name,
                                            dataset=train_tensor_dataset,
                                            variable_names=all_variable_names,
                                            variable_preprocessors=variable_preprocessors,
                                            variable_dtypes=variable_dtypes,
                                            variable_shapes=train_var_shapes) 

            pickle_file_name = 'train_dataset.pt'
            torch.save(ae_train_dataset, dataset_save_dir + '/' + pickle_file_name)

            ae_val_dataset = CreateDataset(name=dataset_name,
                                        dataset=val_tensor_dataset,
                                        variable_names=all_variable_names,
                                        variable_preprocessors=variable_preprocessors,
                                        variable_dtypes=variable_dtypes,
                                        variable_shapes=val_var_shapes) 

            pickle_file_name = 'val_dataset.pt'
            torch.save(ae_val_dataset, dataset_save_dir + '/' + pickle_file_name)

            print(' --> Saved dataset to pickle under /datasets directory.')
    elif mode == 'predict':
        # Save the processed data to .csv file for easy preview
        preview_file_name = 'predict_dataset_preview.csv'
        np.savetxt(dataset_save_dir + '/' + preview_file_name, samples_preview, delimiter=',')

        pickle_file_name = 'predict_dataset.pt'
        torch.save(tensor_dataset, dataset_save_dir + '/' + pickle_file_name)
    else:
        raise ValueError(f'Invalid mode specified {mode}.')
    
def create_kfold_datasets(cross_val_type, dataset, n_splits, global_seed, class_col=None):
    """ Creates kfold datasets from the dataset provided."""

    # Create a list the will store the kfold datasets
    train_tensor_datasets = []
    val_tensor_datasets = []

    # Create a list that will store the variable shapes for each kfold dataset
    train_tensor_datasets_var_shapes = []
    val_tensor_datasets_var_shapes = []
    
    # Create a cross validator
    cross_validator = KFold(n_splits=n_splits, shuffle=True, random_state=global_seed)

    i = 0
    for variable_name, samples in zip(dataset.keys(), dataset.values()):

        # Get the indices for each fold for each variable
        idxs = cross_validator.split(samples)

        for j, (train_idx, test_idx) in enumerate(idxs):

            print(f' --> {variable_name} Fold {j} : train_idx {train_idx.shape} test_idx {test_idx.shape}')

            # In case of the first variable, create the list and dictionary
            if i == 0:
                train_tensor_datasets.extend([{variable_name : torch.tensor(samples[train_idx][:], dtype=torch.float32)}])
                val_tensor_datasets.extend([{variable_name : torch.tensor(samples[test_idx][:], dtype=torch.float32)}])

                train_tensor_datasets_var_shapes.extend([{variable_name : samples[train_idx][:].shape}])
                val_tensor_datasets_var_shapes.extend([{variable_name : samples[test_idx][:].shape}])

            # For all the other variables access the dictionaries for each fold and append the new variable and its samples
            else:
                train_tensor_datasets[j][variable_name] = torch.tensor(samples[train_idx][:], dtype=torch.float32)
                val_tensor_datasets[j][variable_name] = torch.tensor(samples[test_idx][:], dtype=torch.float32)

                train_tensor_datasets_var_shapes[j][variable_name] = samples[train_idx][:].shape
                val_tensor_datasets_var_shapes[j][variable_name] = samples[test_idx][:].shape
        
        i += 1

    return train_tensor_datasets, train_tensor_datasets_var_shapes, val_tensor_datasets, val_tensor_datasets_var_shapes

def create_train_val_datasets(dataset, test_split, global_seed, dataset_save_dir):
    """ Creates a train and test list consisting of the train and test numpy arrays respectively."""

    # Create a dictionary to store the train and val datasets
    train_tensor_dataset = {}
    val_tensor_dataset = {}

    # Create a dictionary to store the variable shapes for each dataset
    train_var_shapes = {}
    val_var_shapes = {}

    i = 0
    for variable_name, samples in zip(dataset.keys(), dataset.values()):

        train_samples, val_samples = train_test_split(samples, test_size=test_split, random_state=global_seed)

        if i == 0:
            train_samples_preview = train_samples
            val_samples_preview = val_samples
        else:
            train_samples_preview = np.hstack((train_samples_preview, train_samples))
            val_samples_preview = np.hstack((val_samples_preview, val_samples))

        #idx = int(0.8*train_val_samples.shape[0])
        #train_samples = train_val_samples[:idx, :]
        #val_samples = train_val_samples[idx:, :]

        train_var_shapes[variable_name] = train_samples.shape
        val_var_shapes[variable_name] = val_samples.shape

        train_tensor_dataset[variable_name] = torch.tensor(train_samples, dtype=torch.float32)
        val_tensor_dataset[variable_name] = torch.tensor(val_samples, dtype=torch.float32)

        i += 1

    # Save the train and val samples preview to a csv
    preview_file_name = 'train_dataset_preview.csv'
    np.savetxt(dataset_save_dir + '/' + preview_file_name, train_samples_preview, delimiter=',')

    preview_file_name = 'val_dataset_preview.csv'
    np.savetxt(dataset_save_dir + '/' + preview_file_name, val_samples_preview, delimiter=',')

    return train_tensor_dataset, train_var_shapes, val_tensor_dataset, val_var_shapes

def weight_samples(dataframe, dataset_dict):
    """ Weight samples based on provided weighting scheme."""

    col_idx = dataset_dict['weight_samples']['col_idx']
    nbins = dataset_dict['weight_samples']['nbins']
    scheme = dataset_dict['weight_samples']['scheme']

    # First check the data type of the dataframe column
    samples = dataframe.iloc[:, col_idx].values

    if is_numeric_dtype(samples):
        print(' --> Dataframe column is of numeric data type')
        print(f' --> Creating a histogram for the numeric sample with {nbins} bins')

        hist, bin_edges = np.histogram(samples, bins=nbins)

        # Righ most bin is hlaf open
        bin_num_for_each_sample = np.digitize(samples, bin_edges, right=False)

        # Get the counts for each bin_num
        bin_nums, bin_num_counts = np.unique(bin_num_for_each_sample, return_counts=True)

        if scheme == 'inv_count':
            wts_for_each_bin = 1/bin_num_counts
        else:
            raise ValueError(f' --> Requested weighting scheme {scheme} is not available.')

        # Assign the sample weights to each sample 
        sample_wts = []
        for bin_num in bin_num_for_each_sample:
            sample_wts.append(wts_for_each_bin[bin_num-1])

        sample_wts = np.array(sample_wts, dtype=np.float32).reshape(-1,1)

        # Create a new column in the dataframe to store the sample 
        #dataframe['sample_wts'] = sample_wts

        return sample_wts

    elif is_string_dtype(samples):
        print(' --> Dataframe column is of string data type.')

        # Get the different classes for the categorical variable and the counts for each class
        value_counts = dataframe.iloc[:, df_col].value_counts()

        counts = value_counts.values
        classes = [cl for cl in value_counts.index.values.tolist()]

        for i, cl in enumerate(classes):
            print(f' --> Found the {cl} class with {counts[i]} counts.')

        # Assign sample weighting to each sample in the class
        print(f' --> Applying the {scheme} weight scheme.')
        sample_weights = []

        # Iterate through the samples
        for sample in samples:
            # Find which class the sample belongs to
            j = classes.index(sample)
            sample_weights.append(generate_sample_weight(scheme, counts[j]))

        return sample_wts

    else:
        raise ValueError('Dataframe contains a data type different from numeric and string.')
    
