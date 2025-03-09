from torch import nn, optim, zeros, equal
from pytorch_lightning.callbacks import ModelCheckpoint

from NestedAE.nn_utils import *
from test_inputs.sample_nn_inputs import sample_nn_params_dict
from test_inputs.sample_train_inputs import sample_nn_train_params_dict

def test_check_dict_key_exists(sample_nn_params_dict):
    assert check_dict_key_exists('modules', sample_nn_params_dict) == True

def test_get_module_input_dim_for_encoder(sample_nn_params_dict):
    module_dict_enc = sample_nn_params_dict['modules']['encoder']
    assert get_module_input_dim(module_dict_enc['connect_to'], sample_nn_params_dict, {'desc1':(100, 15)}) == 15

def test_get_module_input_dim_for_predictor(sample_nn_params_dict):
    module_dict_pred = sample_nn_params_dict['modules']['predictor']
    assert get_module_input_dim(module_dict_pred['connect_to'], sample_nn_params_dict, {'desc1':(100, 15)}) == 10

def test_set_layer_init(sample_nn_params_dict):
    enc_layer_list = [nn.Linear(15, 25),
                      nn.Tanh(),
                      nn.Linear(25, 10),
                      nn.Tanh()]
    module_dict_enc = sample_nn_params_dict['modules']['encoder']
    layer_list_init = set_layer_init(enc_layer_list, module_dict_enc, init='bias')
    assert equal(layer_list_init[0].bias.data, zeros(25)) == True

def test_create_optimizer_object(sample_nn_train_params_dict):
    modules = nn.ModuleDict({'encoder':nn.Linear(15, 10),
                              'predictor':nn.Linear(10, 10),
                              'decoder':nn.Linear(10, 15)})
    assert isinstance(create_optimizer_object(modules, sample_nn_train_params_dict), optim.Adam)

def test_create_scheduler_object(sample_nn_train_params_dict):
    modules = nn.ModuleDict({'encoder':nn.Linear(15, 10),
                              'predictor':nn.Linear(10, 10),
                              'decoder':nn.Linear(10, 15)})
    optimizer = create_optimizer_object(modules, sample_nn_train_params_dict)
    assert isinstance(create_scheduler_object(optimizer, sample_nn_train_params_dict)['scheduler'], optim.lr_scheduler.StepLR)

def test_create_callback_object(sample_nn_train_params_dict):
    ae_save_dir = '.'
    assert isinstance(create_callback_object(sample_nn_train_params_dict, ae_save_dir)[0], ModelCheckpoint)







