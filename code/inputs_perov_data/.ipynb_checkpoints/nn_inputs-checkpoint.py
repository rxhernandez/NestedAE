# Input dim calculation : Total num of feature - Num of encoded features + (Num of categories for each encoded feature)
# Note : Num of categories for each encoded feature = num of one hot encoded categories
# Note : If using label binarizer then that counts as adding only 1 feature column

list_of_nn_params_dict=[
        
        {'model_name':'vanilla_ae_1',
         'input_dim':27 - 5 + (8 + 1 + 4 + 5 + 8),
         'num_hidden_layers':1, # HP
         'num_hidden_nodes_per_layer':20, # HP
         'hidden_layer_activation':'relu', # HP
         'latent_dim':10, # HP
         'latent_layer_activation':'relu', # HP
         'add_supervision_on_latent':False,
         'y_dim':None,
         'y_layer_activation':None,
         'output_layer_activation':'elu', # HP
         'kernel_initializer':'glorot_uniform',
         'bias_initializer':'zeros',
         'kernel_regularizer':None,
         'bias_regularizer':None,
         'activity_regularizer':None},
    
        {'model_name':'vanilla_ae_2',
         'input_dim': 10 - 6 + (5 + 1 + 3 + 7 + 10 + 4),
         'num_hidden_layers':1, # HP
         'num_hidden_nodes_per_layer':20, # HP
         'hidden_layer_activation':'relu', # HP
         'latent_dim':10, # HP
         'latent_layer_activation':'relu', # HP
         'add_supervision_on_latent':False,
         'y_dim':None,
         'y_layer_activation':None,
         'output_layer_activation':'elu', # HP
         'kernel_initializer':'glorot_uniform',
         'bias_initializer':'zeros',
         'kernel_regularizer':None,
         'bias_regularizer':None,
         'activity_regularizer':None}
        
]

