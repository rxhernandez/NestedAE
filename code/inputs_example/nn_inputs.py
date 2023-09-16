"""Neural network inputs dictionary"""

list_of_nn_params_dict=[
    
       # ... NN params for the first autoencoder go here ...
       {

              # Name of the model goes here. A directory with this name will be created in the runs folder.
              'model_type':'model_name_goes_here',

              # You can create block of neural networks (called submodules) and connect them however you like.
              'submodules':{

                     'submodule_1_name_goes_here':{

                            'connect_to':['first_variable_name_of_dataset1_goes_here',
                                          'second_variable_name_of_dataset2_goes_here'], 
                                   # desc : Data that should be passed to submodule. 
                                   #        In case dataset, specify variable name from the dataset_inputs.py file.
                                   #        In case of another submodule, specify the name of the submodule.
                                   # dtype : list

                            'num_nodes_per_layer':[100, 100], 
                                   # desc : Number of nodes in each layer.
                                   # dtype : list

                            'layer_type':['linear', 'linear'], 
                                   # desc: Type of layer. CURRENT SUPPORT ONLY FOR LINEAR LAYERS.
                                   # dtype : list

                            'layer_activation':['relu', 'relu'],
                                   # desc : Activation function for each layer. 
                                   #        Please check set_layer_activation() in nn_utils.py 
                                   #        for supported activations.
                                   # dtype : list

                            'layer_kernel_init':['xavier_normal', 'xavier_normal'], 
                                   # desc : Weight initialization for each layer. 
                                   #        Please check set_layer_init() in nn_utils.py 
                                   #        for supported initializations.
                                   # dtype : list

                            'layer_kernel_init_gain':[1, 1], 
                                   # desc : Gain for weight initialization for each layer.
                                   # dtype : list. 

                            'layer_bias_init':['zeros', 'zeros'],
                                   # desc : Bias initialization for each layer. 
                                   # dtype : list.

                            'layer_weight_reg':{'l1':0.01, 'l2':0},
                                   # desc : Weight regularization.
                                   #        Same regularization will be applied to all layers.
                                   # dtype : dict 

                            'layer_dropout':[{'type':'Dropout', 'p':0.1}, None], 
                                   # desc : Dropout layer. Specify None if no dropout required for layer.
                                   # dtype : list(dict)

                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'bg'}, 
                                   # desc : Loss function. Use only if submodule output is a prediction. 
                                   #        Please check create_loss_object() in nn_utils.py for supported loss functions.
                                   # dtype : dict

                            'load_params':'../runs/model_name_goes_here/submodule_params/submodule_1_name_goes_here_params.pt',
                                   # desc : Load parameters of this submodule from a file.
                                   # dtype : str

                            'save_params':True, 
                                   # desc : Save parameters of this submodule after training.
                                   # dtype: bool

                            'save_output_on_fit_end':True,
                                   # desc :  Save output of this submodule after training.
                                   # dtype : bool

                            'save_output_on_epoch_end':True  
                                   # desc : Save output of this submodule after each epoch.
                                   # dtype : bool
                     },

                     'submodule_2_name_goes_here':{
                     
                           'connect_to':'submodule_1_name_goes_here',
                            # ... params for this submodule go here ...
                     }

              }

       },

       {

              'model_type':'model_name_goes_here',

              'submodules':{
                  
                  # ... submodules for autoencoder 2 go here ...
                  
              }

       }

]


