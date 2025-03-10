""" Utils to create processed dataset and perform k-fold splitting."""
from pandas import read_csv
from numpy import float32 as np_float32
from numpy import hstack, savetxt
from torch import float32 as torch_float32
from torch import load, save, tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold

class TensorDataset(Dataset):
    """ Creates a PyTorch Tensor Dataset. """
    def __init__(self, name, tensor_dictionary,
                 desc_preprocessors):
        self.name = name
        self.tensor_dictionary = tensor_dictionary
        self.desc_preprocessors = desc_preprocessors
        self.desc_names = list(self.tensor_dictionary.keys())
        self.desc_shapes = {}
        self.desc_dtypes = {}
        for (desc_name, desc) in self.tensor_dictionary.items():
            self.desc_shapes[desc_name] = list(desc.shape)
            self.desc_dtypes[desc_name] = desc.dtype
        # Get all the desc shapes
        desc_shapes_list = list(self.desc_shapes.values())
        num_samples = desc_shapes_list[0][0]
        sum_desc_dim = sum([shape[1] for shape in desc_shapes_list]) 
        self.shape = (num_samples, sum_desc_dim)

    # Returns the number of samples in each dataset
    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        item = {}
        for desc_name in self.desc_names:
            item[desc_name] = self.tensor_dictionary[desc_name][idx, :]
        return item

def preprocess_datasets(dataset_save_dir_path, nn_datasets_dict, dataset_type):
    """ Reads data from specified source and processes it according to specifications in nn_dataset_dict

    Args:
        dataset_save_dir_path: Model directory 
        nn_datasets_dict: dictionary in dataset_inputs.py file.
        dataset_type : Which datasets to use (train, predict, test ...) (top-level key in nn_datasets_dict)

    Returns:
        processed_dataset (dict) : Dictionary containing all desciptor samples. (Key : desc_name, Value : samples)
        desc_preprocessors (dict) : Dictionary containing all the preprocessors for descriptors
    """
    datasets = nn_datasets_dict[dataset_type]
    print(f" --> Working with '{dataset_type}' dataset types.")

    # Create a combined dataset that stores all datasets provided in nn_datasets_dict
    processed_dataset = {}
    processed_tensor_dataset = {}
    desc_preprocessors = {}

    # Outer loop iterates over each dataset (Ex : X, y, latents ...) belonging to the dataset_type.
    for i, (dataset_name, dataset_dict) in enumerate(datasets.items()):
        dataset_file_path = dataset_dict['path']
        dataset_file_name = dataset_file_path.split('/')[-1]
        dataset_file_type = dataset_file_name.split('.')[-1]
        if dataset_file_type != 'csv':
            raise FileNotFoundError(' --> Supported file type not found')
        else:
            print(f' --> Found .csv file : {dataset_file_name}')
            try:
                header = dataset_dict['header']
                skiprows = dataset_dict['skiprows']
            except:
                header = 0
                skiprows = None
            dataframe = read_csv(dataset_file_path, header=header,
                                skiprows=skiprows)
        print(f' --> Loaded {dataset_name} dataset as a dataframe.')
        print(f' --> {dataset_name} Dataframe shape : {dataframe.shape}')
        print(' --> Dataframe head.')
        print(dataframe.head())

        # Inner loop iterates over each descriptor in the dataset
        for j, (descriptor_name, descriptor_dict) in enumerate(dataset_dict['descriptors'].items()):
            cols = descriptor_dict['cols']
            cols = cols if not isinstance(cols, list) else list(cols)
            # Check for any NA values in descriptor
            na_values = dataframe.iloc[:, cols].isna()
            if na_values.any().any():
                raise ValueError(f' --> NA values found in {descriptor_name} \
                        dataframe. Check log file for details.')
            else:
                print(f' --> No NA values found in {descriptor_name}.')

            # Extract samples for descriptor
            samples = dataframe.iloc[:, cols].values
            if samples.shape[1] == 1:
                samples = samples.reshape(-1, 1)
            print(f' --> Data for {descriptor_name} from {dataset_name} dataframe cols {cols}.')
            
            try:
                load_preprocessor = dataset_dict['load_preprocessor']
            except:
                load_preprocessor = False
            
            if load_preprocessor:
                print(f' --> Loading preprocessor for {descriptor_name} from {dataset_save_dir_path}/combined_train_datasets.pt.')
                loaded_dataset = load(dataset_save_dir_path + '/combined_train_datasets.pt', weights_only=False)
                preprocessor = loaded_dataset.desc_preprocessors[descriptor_name]
                if preprocessor is not None:
                    if isinstance(preprocessor, StandardScaler):
                        samples = samples.astype(np_float32)
                        samples = preprocessor.transform(samples)
                    if isinstance(preprocessor, OneHotEncoder):
                        samples = preprocessor.transform(samples).toarray()
            else:
                preprocess_scheme = descriptor_dict['preprocess']
                if preprocess_scheme is None:
                    samples = samples.astype(np_float32)
                    print(f' --> No preprocessing done for {descriptor_name} \
                                from {dataset_name} dataframe cols {cols}.')
                    preprocessor, dtype, shape = None, samples.dtype, samples.shape
                    print(f' --> {descriptor_name} : ({dtype}, {shape}) ')
                if preprocess_scheme == 'std':
                    samples = samples.astype(np_float32)
                    _ss = StandardScaler()
                    samples, preprocessor = _ss.fit_transform(samples), _ss
                    print(f' --> {descriptor_name} dtype : {samples.dtype}, shape : {samples.shape}')
                if preprocess_scheme == 'ohe':
                    _ohe = OneHotEncoder(dtype=np_float32)
                    _ohe.fit(samples)
                    samples, preprocessor, categories = _ohe.transform(samples).toarray(), _ohe, _ohe.categories_[0]
                    print(f' --> {descriptor_name} dtype : {samples.dtype}, shape : {samples.shape}')
                    print(f' --> Encoded col {cols} \
                        with {len(categories)} \
                        categories {categories}')

            if i == 0 and j == 0:
                samples_preview = samples
            else:
                samples_preview = hstack((samples_preview, samples))
            
            processed_dataset[descriptor_name] = samples
            processed_tensor_dataset[descriptor_name] = tensor(samples, dtype=torch_float32)
            desc_preprocessors[descriptor_name] = preprocessor
    
    dataset_name = f'combined_{dataset_type}_datasets'
    tensor_dataset = TensorDataset(name=dataset_name,
                                   tensor_dictionary=processed_tensor_dataset,
                                   desc_preprocessors=desc_preprocessors)
    # with open(dataset_save_dir_path + f'/{dataset_name}.pkl', 'wb') as f:
    #     pickle.dump(tensor_dataset, f)
    save(tensor_dataset, dataset_save_dir_path + f'/{dataset_name}.pt') 
    savetxt(dataset_save_dir_path + f'/{dataset_name}_preview.csv', samples_preview, 
            delimiter=',', header=','.join(list(processed_tensor_dataset.keys())))
    print(f' --> Saved combined dataset to disk at {dataset_save_dir_path}/{dataset_name}.pt')
    return processed_dataset, desc_preprocessors

def create_kfold_datasets(processed_dataset, desc_preprocessors, dataset_save_dir_path='.', n_splits=2, seed=100):
    """ Creates kfold datasets from the dataset dictionary provided.
    
    Args:
        processed_dataset (dict): Dictionary with descriptors as keys and samples as variables.
        desc_preprocessors (dict): Dictionary containing all the preprocessors for descriptors
        dataset_save_dir_path (str): Path to store TorchDataset. Defaults to current directory.
        n_splits (int): Number of splits to create. Defaults to 2.
        seed (int): Random seed. Defaults to 100.
    
    Returns: None
    """
    # Create a kfold cross validator
    cross_validator = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Get the first variable out of the dataset to get the train and test idxs
    desc_names = list(processed_dataset.keys())
    first_variable_samples = processed_dataset[desc_names[0]]
    idxs = cross_validator.split(first_variable_samples)
    # Iterate over the folds
    train_tensor_dictionary = {}
    val_tensor_dictionary = {}
    # Iterate over the folds
    for i, (train_idx, val_idx) in enumerate(idxs):
        print(f' --> Fold {i} : train_idx {train_idx.shape} val_idx {val_idx.shape}')
        # Store the train and val datasets for each fold
        for j, desc_name in enumerate(desc_names):
            train_samples = processed_dataset[desc_name][train_idx][:]
            val_samples = processed_dataset[desc_name][val_idx][:]
            if j == 0:
                train_samples_preview = train_samples
                val_samples_preview = val_samples
            else:
                train_samples_preview = hstack((train_samples_preview, train_samples))
                val_samples_preview = hstack((val_samples_preview, val_samples))
            train_tensor_dictionary[desc_name] = tensor(train_samples, dtype=torch_float32)
            val_tensor_dictionary[desc_name] = tensor(val_samples, dtype=torch_float32)
        # Store the tensor dataset for fold as a TensorDataset for easy training
        train_tensor_dataset = TensorDataset(name=f'train_fold_{i}',
                                             tensor_dictionary=train_tensor_dictionary,
                                             desc_preprocessors=desc_preprocessors)
        val_tensor_dataset = TensorDataset(name=f'val_fold_{i}', 
                                           tensor_dictionary=val_tensor_dictionary,
                                           desc_preprocessors=desc_preprocessors)
        # Save the tensor datasets
        # with open(dataset_save_dir_path + f'/train_fold_{i}.pkl', 'wb') as f:
        #     pickle.dump(train_tensor_dataset, f)
        save(train_tensor_dataset, dataset_save_dir_path + f'/train_fold_{i}.pt') 
        savetxt(dataset_save_dir_path + '/' + f'train_fold_{i}_preview.csv',
                   train_samples_preview, delimiter=',', header=','.join(list(processed_dataset.keys())))
        
        # with open(dataset_save_dir_path + f'/val_fold_{i}.pkl', 'wb') as f:
        #     pickle.dump(val_tensor_dataset, f)
        save(val_tensor_dataset, dataset_save_dir_path + f'/val_fold_{i}.pt')
        savetxt(dataset_save_dir_path + '/' + f'val_fold_{i}_preview.csv',
                   val_samples_preview, delimiter=',', header=','.join(list(processed_dataset.keys())))
        print(f' --> Saved {i}-fold datasets to {dataset_save_dir_path} directory.')

