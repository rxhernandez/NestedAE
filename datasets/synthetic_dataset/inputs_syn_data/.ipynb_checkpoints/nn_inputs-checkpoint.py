list_of_nn_params_dict=[
        
        {'model_name':'vanilla_ae_1',
         'num_hidden_layers':1, 
         'num_hidden_nodes_per_layer':30, 
         'hidden_layer_activation':'elu',
         'latent_dim':10, 
         'latent_layer_activation':'elu',
         'add_supervision_on_latent':False, 
         'y_layer_activation':None,
         'X_hat_layer_activation':['elu'],
         'kernel_initializer':'glorot_uniform',
         'bias_initializer':'zeros',
         'kernel_regularizer':None,
         'bias_regularizer':None,
         'activity_regularizer':None},
    
        {'model_name':'vanilla_ae_2',
         'num_hidden_layers':1,
         'num_hidden_nodes_per_layer':20,
         'hidden_layer_activation':'elu',
         'latent_dim':10,
         'latent_layer_activation':'elu',
         'add_supervision_on_latent':False,
         'y_layer_activation':None,
         'X_hat_layer_activation':['elu'],
         'kernel_initializer':'glorot_uniform',
         'bias_initializer':'zeros',
         'kernel_regularizer':None,
         'bias_regularizer':None,
         'activity_regularizer':None}
        
]

