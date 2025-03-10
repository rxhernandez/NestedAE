"""Neural network inputs dictionary"""

from torch import nn

# Can define custom losses and metrics and pass them as arguments to the loss and metric keys in the dictionary
#### Custom Losses and Metrics ####


####################################

# Hyperparam tuning of autoencoders done by mirroring the encoder architecture for the decoders and predictors.
# For any submodule other than encoder and if making a prediction must specify the 'output_dim', 'output_activation', and 'loss' keys.

list_of_nn_params_dict=[
    
       # ... NN params for the first autoencoder go here ...
       {
              'name':'ae1',
              'modules':{

                     'encoder':{
                            # 'connect_to':['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sb', 'Pb', 'Cl', 'Br', 'I'], # List of features to connect to in database go here
                            'connect_to':['A_comp', 'B_comp', 'X_comp'],
                            'hidden_dim':50,
                            'hidden_layers':1,
                            'hidden_activation':nn.ReLU(),
                            'output_dim':10,
                            # 'output_dim':{'values':[8, 10, 12, 14]},
                            # 'hidden_dim':{'values':[25, 50, 100, 200]}, # Use the 'values' key for hyperparam tuning
                            # 'hidden_layers':{'values':[1, 2, 3]},
                            # 'hidden_activation':{'values':['tanh', 'relu']},
                            # 'output_activation':nn.ReLU(),
                            'layer_type':'linear', 
                            'layer_kernel_init':'xavier_normal',
                            'layer_weight_reg_l2':0.001,
                            'param_optimization':False,
                            'save_output_on_fit_end':True
                     },

                     'predictor':{
                            'connect_to':['encoder'],
                            'hidden_dim':50,
                            'hidden_layers':1,
                            'hidden_activation':nn.ReLU(),
                            'output_dim':1,
                            'layer_type':'linear',
                            'layer_kernel_init':'xavier_normal',
                            'layer_weight_reg_l2':0.001,
                            'loss':{'type':nn.L1Loss(),
                                   'wt':1,
                                   'target':'bg'},
                            'param_optimization':False,
                            'save_output_on_fit_end':True
                     },

                     'A_comp_decoder':{
                            'connect_to':['encoder'],
                            'hidden_dim':50,
                            'hidden_layers':1,
                            'hidden_activation':nn.ReLU(),
                            'output_dim':5,
                            # 'output_activation':nn.Softmax(),
                            'layer_type':'linear',
                            'layer_kernel_init':'xavier_normal',
                            'layer_weight_reg_l2':0.001,
                            # 'loss':{'type':nn.KLDivLoss(reduction='batchmean'),
                            #        'wt':1,
                            #        'target':'A_comp'},
                            'loss':{'type':nn.L1Loss(),
                                   'wt':1,
                                   'target':'A_comp'},
                            # 'metric':[nn.L1Loss()],
                            'param_optimization':False,
                            'save_output_on_fit_end':True
                     },

                     'B_comp_decoder':{
                            'connect_to':['encoder'],
                            'hidden_dim':50,
                            'hidden_layers':1,
                            'hidden_activation':nn.ReLU(),
                            'output_dim':6,
                            # 'output_activation':nn.Softmax(),
                            'layer_type':'linear',
                            'layer_kernel_init':'xavier_normal',
                            'layer_weight_reg_l2':0.001,
                            # 'loss':{'type':nn.KLDivLoss(reduction='batchmean'),
                            #        'wt':1,
                            #        'target':'B_comp'},
                            'loss':{'type':nn.L1Loss(),
                                   'wt':1,
                                   'target':'B_comp'},
                            # 'metric':[nn.L1Loss()],
                            'param_optimization':False,
                            'save_output_on_fit_end':True
                     },

                     'X_comp_decoder':{
                            'connect_to':['encoder'],
                            'hidden_dim':50,
                            'hidden_layers':1,
                            'hidden_activation':nn.ReLU(),
                            'output_dim':3,
                            # 'output_activation':nn.Softmax(),
                            'layer_type':'linear',
                            'layer_kernel_init':'xavier_normal',
                            'layer_weight_reg_l2':0.001,
                            # 'loss':{'type':nn.KLDivLoss(reduction='batchmean'),
                            #        'wt':1,
                            #        'target':'X_comp'},
                            'loss':{'type':nn.L1Loss(),
                                   'wt':1,
                                   'target':'X_comp'},
                            # 'metric':[Accuracy(task='MULTILABEL', num_labels=3)],
                            # 'metric':[nn.L1Loss()],
                            'param_optimization':False,
                            'save_output_on_fit_end':True
                     }
              }

       }

]