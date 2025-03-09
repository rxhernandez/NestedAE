import pytest
from torch import nn
from torchmetrics import MeanAbsoluteError

'''
* This sample_nn_params_dict lists the dictionary
structure required for creating the autoencoder model.

* The encoder module shows all the supported 'keys' for
defining the module dictionary.

* In case hyperparameter tuning is required for module,
then specify the values to test by using a dictionary 
with 'values' as the key. An example is shown below :

    'hidden_dim':{'values':[25, 50]}

* To 'mirror' the encoder's hyperparameter for the 
predictor and decoder, simply type 'mirror' for the
respective module key. 

    {'name':'test',
    'modules':{
        'encoder':{
            'hidden_dim':{'values':[25, 50]}
            },

        'predictor':{
            'hidden_dim':'mirror'
        }}
    }
    
'''

@pytest.fixture
def sample_nn_params_dict():
    return {'name':'test', 
                # Required
                # Desc : Name of the model
                # Supported input type : str

            'modules':{
                # Required
                # Desc : Dictionary of modules
                # Supported input type : dict

                'encoder':{
                    'connect_to':['desc1'], 
                        # Required
                        # Desc : List of features to connect to in database go here
                        # Supported input type : list
                    'output_dim':10, 
                        # Required
                        # Desc : Number of nodes in output layer
                        # Supported input type : int
                    'hidden_dim':25, 
                        # Optional (default is None)
                        # Desc : Number of nodes in hidden layer. If no hidden layer, set to None
                        # Supported input type : int/None
                    'hidden_layers':1, 
                        # Optional (default is 0)
                        # Desc : Number of hidden layers. If no hidden layer, set to None
                        # Supported input type : int/None
                    'hidden_activation':nn.Tanh(), 
                        # Optional (default is None)
                        # Desc : Activation function for each layer. 
                        #        Please check set_layer_activation() in nn_utils.py 
                        #        for supported activations.
                        # Supported input type : to
                    'output_activation':nn.Tanh(), 
                        # Optional (default is None)
                        # Desc : Activation function for each layer. 
                        #        Please check set_layer_activation() in nn_utils.py 
                        #        for supported activations.
                        # Supported input type : str
                    'layer_type':'linear', 
                        # Optional (default is linear)
                        # Desc : Type of layer. CURRENT SUPPORT ONLY FOR LINEAR LAYERS.
                        # Supported input type : str
                    'layer_kernel_init':'xavier_normal', 
                        # Optional (default is U(-k^0.5, k^0.5) where k = 1/sqrt(fan_in)) 
                        # Desc : Weight initialization for each layer. 
                        #        Please check set_layer_init() in nn_utils.py 
                        #        for supported initializations.
                        # Supported input type : str
                    'layer_bias_init':'zeros', 
                        # Optional (default is U(-k^0.5, k^0.5) where k = 1/sqrt(fan_in))
                        # Desc : Bias initialization for each layer. 
                        #        Please check set_layer_init() in nn_utils.py 
                        #        for supported initializations.
                        # Supported input type : str
                    'layer_weight_reg_l1':0.0, 
                        # Optional (default is 0) 
                        # Desc : L1 Weight regularization.
                        # Supported input type : float
                    'layer_weight_reg_l2':0.001, 
                        # Optional (default is 0) 
                        # Desc : L2 Weight regularization.
                        # Supported input type : float
                    'layer_dropout':nn.Dropout(p=0.5),
                        # Optional (default is None) 
                        # Desc : Dropout after each layer
                        #        Please check set_layer_dropout() in nn_utils.py
                        #        for supported dropout types.
                        # Supported input type : float
                    'save_output_on_fit_end':False, 
                        # Optional (default is False) 
                        # Desc : Save the output of the model on entire dataset at end of training
                        # Supported input type : bool
                    'save_output_on_epoch_end':False, 
                        # Optional (default is False)
                        # Desc : Save the output of the model on entire dataset at end of each epoch
                        # Supported input type : bool
                    'save_params':True, 
                        # Optional (default is False)
                        # Desc : Save model params to a file
                        # Supported input type : bool
                    # 'load_params':False 
                    #     # Optional (default is False) 
                    #     # Desc : Load model params from a file
                    #     # Supported input type : bool
                    },

                'predictor':{
                    'connect_to':['encoder'],
                    'output_dim':1,
                    'output_activation':nn.ReLU(),
                    'hidden_dim':25,
                    'hidden_layers':1,
                    'loss':{'type':nn.L1Loss(), 
                            'wt':1,
                            'target':'target'},
                    # Optional
                    # Desc : Loss dictionary
                    # Supported input type : dict
                    'metric':[MeanAbsoluteError()],
                    # Optional
                    # Desc : Type of metric to use for reporting
                    # Supported input type : list
                           },

                'decoder':{
                    'connect_to':['encoder'],
                    'output_dim':15,
                    'hidden_dim':25,
                    'hidden_layers':1,
                    'hidden_activation':nn.Tanh(),
                    'loss':{'type':nn.L1Loss(),
                           'wt':1,
                           'target':'desc1'}}
                }
            }