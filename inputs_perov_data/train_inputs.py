"""Neural network training inputs dictionary"""

list_of_nn_train_params_dict=[
        
        {'global_seed':100, 
         'epochs':500,  
         'batch_size':1, 
         'shuffle_data_between_epochs':True, 
         'val_split':0.1, 
         'optimizer':{'type':'adam', 'module_name':['encoder', 'predictor', 'A_comp_decoder', 'B_comp_decoder', 'X_comp_decoder'], 'lr':[1e-3, 1e-3, 1e-3, 1e-3, 1e-3]},
         'callbacks':{'model_checkpoint':{'monitor':'total_val_loss',
                                          'save_top_k':1,
                                          'mode':'min'}}
         }

]
