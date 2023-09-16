"""Neural network training inputs dictionary"""

list_of_nn_train_params_dict=[
        
        # ... train params for second autoencoder go here ...
        {'global_seed':0, 
            # desc: Set the global random seed
            # dtype: int

         'epochs':1000,  
            # desc: Number of epochs
            # dtype: int

         'batch_size':10, 
            # desc: Batch size
            # dtype: int

         'shuffle_data_between_epochs':True, 
            # desc: Shuffle data between epochs
            # dtype: bool

         'test_split':0.2, 
            # desc: Fraction of data to use for validation
            # dtype: float

         'optimizer':{'type':'adam', 'lr':[0.001, 0.001], 'submodule_name':['submodule_1_name_goes_here', 'submodule_2_name_goes_here']},

         #'scheduler':{'type':'reduce_lr_on_plateau'},
         #'scheduler':{'type':'multi_step', 'milestones':[30], 'gamma':0.1},
         
         'callbacks':{'model_checkpoint':{'monitor':'total_val_loss',
                                          'save_top_k':1,
                                          'mode':'min'}}
         },

         {
             # ... train params for second autoencoder go here ...
         }

]
