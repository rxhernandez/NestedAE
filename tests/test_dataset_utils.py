from torch import tensor
from torch import float32 as torch_float32
from numpy import array, round
from numpy import float32 as np_float32
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from NestedAE.dataset_utils import TensorDataset, preprocess_datasets, create_kfold_datasets
from test_inputs.sample_dataset_inputs import sample_nn_datasets_dict

def test_TensorDataset():
    tensor_dictionary = {'desc1':tensor([[1], [2], [3], [4]], dtype=torch_float32),
                         'desc2':tensor([[5], [6], [7], [8]], dtype=torch_float32),
                         'desc3':tensor([[9], [10], [11], [12]], dtype=torch_float32)}
    desc_preprocessors = {'desc1':None, 'desc2':None, 'desc3':None}
    tensor_dataset = TensorDataset(name='test', tensor_dictionary=tensor_dictionary, 
                                    desc_preprocessors=desc_preprocessors)
    assert tensor_dataset.name == 'test'
    assert tensor_dataset.tensor_dictionary == tensor_dictionary
    assert tensor_dataset.desc_preprocessors == {'desc1':None, 'desc2':None, 'desc3':None}
    assert tensor_dataset.desc_names == ['desc1', 'desc2', 'desc3']
    assert tensor_dataset.desc_shapes == {'desc1':[4, 1], 'desc2':[4, 1], 'desc3':[4, 1]}
    assert tensor_dataset.desc_dtypes == {'desc1':torch_float32, 'desc2':torch_float32, 'desc3':torch_float32}
    assert tensor_dataset.shape == (4, 3)
    assert len(tensor_dataset) == 4
    assert tensor_dataset[0] == {'desc1':tensor([1.]), 'desc2':tensor([5.]), 'desc3':tensor([9.])}

def test_preprocess_datasets_for_train_datasets(sample_nn_datasets_dict):
    # Correct processed dataset for sample_train_dataset
    correct_processed_dataset = {
        'desc1' : array([[1.], [1.], [1.], [1.], [1.]], dtype=np_float32),
        'desc2' : array([[-1.4142], [-0.7071], [0.], [0.7071], [1.4142]], dtype=np_float32),
        'desc3' : array([[1., 0.], [0., 1.], [1., 0.], [1., 0.], [1., 0.]], dtype=np_float32)
    }
    dataset_save_dir_path = 'tests/test_datasets'
    dataset_type = 'train'
    processed_dataset, desc_preprocessors = preprocess_datasets(dataset_save_dir_path, sample_nn_datasets_dict, dataset_type)
    processed_dataset_rounded = {}
    for desc_name in (processed_dataset.keys()):
        processed_dataset_rounded[desc_name] = round(processed_dataset[desc_name], 4)
    for desc_name in (processed_dataset.keys()):
        assert (processed_dataset_rounded[desc_name] == correct_processed_dataset[desc_name]).all()
        if desc_name == 'desc1':
            assert desc_preprocessors[desc_name] == None
        elif desc_name == 'desc2':
            assert isinstance(desc_preprocessors[desc_name], StandardScaler)
        elif desc_name == 'desc3':
            assert isinstance(desc_preprocessors[desc_name], OneHotEncoder)

def test_preprocess_datasets_for_test_datasets(sample_nn_datasets_dict):
    correct_processed_dataset = {
        'desc1' : array([[1.], [1.], [1.]], dtype=np_float32),
        'desc2' : array([[-1.4142], [-0.7071], [0.]], dtype=np_float32),
        'desc3' : array([[1., 0.], [1., 0.], [1., 0.]], dtype=np_float32)
    }
    dataset_save_dir_path = 'tests/test_datasets'
    dataset_type = 'test'
    processed_dataset, _ = preprocess_datasets(dataset_save_dir_path, sample_nn_datasets_dict, dataset_type)
    processed_dataset_rounded = {}
    for desc_name in (processed_dataset.keys()):
        processed_dataset_rounded[desc_name] = round(processed_dataset[desc_name], 4)
    for desc_name in (processed_dataset.keys()):
        assert (processed_dataset_rounded[desc_name] == correct_processed_dataset[desc_name]).all()

def test_create_kfold_datasets(sample_nn_datasets_dict):
    dataset_save_dir_path = 'tests/test_datasets'
    dataset_type = 'train'
    processed_dataset, desc_preprocessors = preprocess_datasets(dataset_save_dir_path, 
                                                                sample_nn_datasets_dict,
                                                                dataset_type)
    create_kfold_datasets(processed_dataset, desc_preprocessors,
                           dataset_save_dir_path=dataset_save_dir_path)
    
