"""Neural network training inputs dictionary"""

list_of_nn_train_params_dict=[
        
        {'global_seed':0,
         'epochs':1500,
         'batch_size':10,
         'shuffle_data_between_epochs':True,
         'test_split':0.1,
         'optimizer':{'type':'adam', 'lr':0.001},
         'callbacks':{'model_checkpoint':{'monitor':'total_val_loss',
                                          'save_top_k':1,
                                          'mode':'min'}}
         },

         {'global_seed':0,
         'epochs':1500,
         'batch_size':100,
         'shuffle_data_between_epochs':True,
         'test_split':0.1,
         'optimizer':{'type':'adam', 'lr':0.001},
         'callbacks':{'model_checkpoint':{'monitor':'total_val_loss',
                                          'save_top_k':1,
                                          'mode':'min'}}
         }

]
