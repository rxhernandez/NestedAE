list_of_nn_train_params_dict=[
        
        {'epochs':500,
         'batch_size':1,
         'use_shuffle':True,
         'callback_list': ['model_checkpoint', 
                           'early_stopping',
                           'csv_log']},
    
        {'epochs':500,
         'batch_size':1,
         'use_shuffle':True,
         'callback_list': ['model_checkpoint', 
                           'early_stopping',
                           'csv_log']},

]
