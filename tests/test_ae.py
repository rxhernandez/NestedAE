from torch import tensor, nn
from torch import float32 as torch_float32

from NestedAE.ae import AE
from test_inputs.sample_nn_inputs import sample_nn_params_dict
from test_inputs.sample_train_inputs import sample_nn_train_params_dict

def test_AE(sample_nn_params_dict, sample_nn_train_params_dict):
    ae_save_dir_path = 'tests'
    dataset_path = 'tests/test_datasets/combined_train_datasets.pt'
    ae = AE(ae_save_dir_path, sample_nn_params_dict, sample_nn_train_params_dict, dataset_path)
    correct_encoder_layer_list = nn.ModuleList([nn.Linear(1, 25), nn.Tanh(), nn.Dropout(p=0.1), nn.Linear(25, 10), nn.Tanh()])
    correct_predictor_layer_list = nn.ModuleList([nn.Linear(10, 25), nn.Linear(25, 1), nn.ReLU()])
    correct_decoder_layer_list = nn.ModuleList([nn.Linear(1, 25), nn.Tanh(), nn.Linear(25, 15)])
    assert ae.name == 'test'
    assert ae.ae_save_dir_path == 'tests'
    assert ae.nn_params_dict == sample_nn_params_dict
    assert ae.nn_train_params_dict == sample_nn_train_params_dict
    assert isinstance(ae.ae_modules['encoder'], nn.ModuleList)
    assert len(ae.ae_modules['encoder']) == 5
    assert isinstance(ae.ae_modules['predictor'], nn.ModuleList)
    assert len(ae.ae_modules['predictor']) == 3
    assert isinstance(ae.ae_modules['decoder'], nn.ModuleList)
    assert len(ae.ae_modules['decoder']) == 3
    assert len(ae.ae_modules) == 3

