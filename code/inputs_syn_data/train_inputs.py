################################################################################################
# Training params for synthetic dataset
################################################################################################

list_of_nn_train_params_dict=[

        {
            
            'global_seed':0,
            'epochs':2000,
            'batch_size':10,
            'shuffle_data_between_epochs':True,
            'test_split':0.2,
            'optimizer':{'type':'adam', 'lr':0.0001},
            'callbacks':{'early_stopping':{'monitor':'total_val_loss',
                                            'min_delta':0.00001,
                                            'patience':200,
                                            'mode':'min'},

                        'model_checkpoint':{'monitor':'total_val_loss',
                                            'save_top_k':1,
                                            'mode':'min'}}

        },

        {

            'global_seed':0,
            'epochs':2000,
            'batch_size':10,
            'shuffle_data_between_epochs':True,
            'test_split':0.2,
            'optimizer':{'type':'adam', 'lr':0.01},
            'callbacks':{'early_stopping':{'monitor':'total_val_loss',
                                            'min_delta':0.0001,
                                            'patience':150,
                                            'mode':'min'},

                        'model_checkpoint':{'monitor':'total_val_loss',
                                            'save_top_k':1,
                                            'mode':'min'}}
                                            
        }

]
