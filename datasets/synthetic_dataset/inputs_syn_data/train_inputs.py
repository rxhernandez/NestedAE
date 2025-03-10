################################################################################################
# Training params for synthetic dataset
################################################################################################

list_of_nn_train_params_dict=[

        {
            'global_seed':0,
            'epochs':1000,
            'batch_size':10,
            'shuffle_data_between_epochs':True,
            'optimizer':{'type':'adam', 'lr':0.001},
            'kfolds':5,
            'callbacks':{'model_checkpoint':{'monitor':'total_val_loss',
                                          'save_top_k':1,
                                          'mode':'min'}}

        },

        {

            'global_seed':0,
            'epochs':1000,
            'batch_size':10,
            'shuffle_data_between_epochs':True,
            'optimizer':{'type':'adam', 'lr':0.001},
            'kfolds':5,
            'callbacks':{'model_checkpoint':{'monitor':'total_val_loss',
                                          'save_top_k':1,
                                          'mode':'min'}}
                                            
        }

]
