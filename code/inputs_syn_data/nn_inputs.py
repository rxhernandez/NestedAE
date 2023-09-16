# import tensorflow as tf
# list_of_nn_params_dict=[
        
#         {'model_name':'vanilla_ae_1',
#          'num_hidden_layers':2, 
#          'num_hidden_nodes_per_layer':[4,3], 
#          'hidden_layer_activation':['elu','elu'],
#          'latent_dim':2, 
#          'latent_layer_activation':'elu',
#          'add_supervision_on_latent':False, 
#          'y_layer_activation':None,
#          'X_hat_layer_activation':['elu'],
#          'kernel_initializer':'glorot_uniform',
#          'bias_initializer':'zeros',
#          'l1_regularization_parameter':0,
#          'l2_regularization_parameter':0.0001,         
#          'bias_regularizer':None,
#          'activity_regularizer':None},
    
#         {'model_name':'vanilla_ae_2',
#          'num_hidden_layers':1,
#          'num_hidden_nodes_per_layer':[20],
#          'hidden_layer_activation':['elu'],
#          'latent_dim':2,
#          'latent_layer_activation':'elu',
#          'add_supervision_on_latent':False,
#          'y_layer_activation':None,
#          'X_hat_layer_activation':['elu'],
#          'kernel_initializer':'glorot_uniform',
#          'bias_initializer':'zeros',
#          'l1_regularization_parameter':0.001,
#          'l2_regularization_parameter':0.001,         
#          'bias_regularizer':None,
#          'activity_regularizer':None}
# ]


# Reconstruction modules must be defined in the way inputs are fed to the model
# Any other outputs to be extracted from the module must be added before the reconstruction outputs

# Dictionary for running a nestedAE model

# If using a variational autoencoder then the submodules have fixed names
# Resampling strategy using normal dictribution : 'mu', 'logvar'

################################################################################################
# Synthetic Database dictionary
################################################################################################

list_of_nn_params_dict=[

       {
              #'model_type':'encoder_l_3_tanh_l1_1em2_decoder_None_no_l1_corr_coefs_seed_0_lr_1em2_bs_180_mae',
              'model_type':'encoder_FAddNoise_ae1',


              'submodules':{

                     'encoder':{

                            'connect_to':'f1tof4',
                            'num_nodes_per_layer':[4,3,2],
                            'layer_type':['linear','linear','linear'],
                            'layer_activation':['tanh','tanh',None],
                            'layer_kernel_init':['xavier_uniform','xavier_uniform','xavier_uniform'],
                            'layer_kernel_init_gain':[1,1,1],
                            'layer_bias_init':['zeros','zeros','zeros'],
                            'layer_weight_reg':{'l1':0.001, 'l2':0},
                            'save_output_on_train_end':True,
                            'save_params':True
                     },

                     'decoder':{

                            'connect_to':'encoder',
                            'num_nodes_per_layer':[2,3,4],
                            'layer_type':['linear','linear','linear'],
                            'layer_activation':[None,'tanh','tanh'],
                            'layer_kernel_init':['xavier_uniform','xavier_uniform','xavier_uniform'],
                            'layer_kernel_init_gain':[1,1,1],
                            'layer_bias_init':['zeros','zeros','zeros'],
                            'layer_weight_reg':{'l1':0.001, 'l2':0},
                            'loss':{'type':'mae',
                                    'wt':1,
                                    'target':'f1tof4'},
                     }

              }

       },

       {
              'model_type':'encoder_FAddNoise_ae2',

              'submodules':{

                     'encoder':{

                            'connect_to':['f5tof13andae1l'],
                            'num_nodes_per_layer':[11,8,6,5],
                            'layer_type':['linear','linear','linear','linear'],
                            'layer_activation':['tanh','tanh','tanh',None],
                            'layer_kernel_init':['xavier_uniform','xavier_uniform','xavier_uniform','xavier_uniform'],
                            'layer_kernel_init_gain':[1,1,1,1],
                            'layer_bias_init':['zeros','zeros','zeros','zeros'],
                            'layer_weight_reg':{'l1':0.001, 'l2':0},
                            'save_output_on_train_end':True,
                            'save_params':True
                     },

                     'decoder':{

                            'connect_to':'encoder',
                            'num_nodes_per_layer':[5,6,8,11],
                            'layer_type':['linear','linear','linear','linear'],
                            'layer_activation':[None,'tanh','tanh','tanh'],
                            'layer_kernel_init':['xavier_uniform','xavier_uniform','xavier_uniform','xavier_uniform'],
                            'layer_kernel_init_gain':[1,1,1,1],
                            'layer_bias_init':['zeros','zeros','zeros','zeros'],
                            'layer_weight_reg':{'l1':0.001, 'l2':0},
                            'loss':{'type':'mae',
                                    'wt':1,
                                    'target':'f5tof13andae1l'},
                     }

              }
       }

]