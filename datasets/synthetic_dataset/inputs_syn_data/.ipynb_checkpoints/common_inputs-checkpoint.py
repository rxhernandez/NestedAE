# Parameters for Callbacks

# List of dictionaries containing common parameters to apply to all ae's
list_of_common_params_dict = [
       
       {'global_seed':4004},
        
       # Dictionary of callback inputs
       {'model_checkpoint':{'monitor':'val_mean_total_loss',
                            'verbose':0,
                            'save_best_only':True,
                            'save_weights_only':True,
                            'mode':'auto',
                            'save_freq':'epoch',
                            'initial_value_threshold':None},
        
        'early_stopping':{'monitor':'val_mean_total_loss',
                          'min_delta':1e-4,
                          'patience':20,
                          'verbose':1,
                          'mode':'auto',
                          'baseline':None,
                          'restore_best_weights':True},
        
        'csv_log':{'filename':'epoch_results.csv',
                   'separator':',',
                   'append':False}}

] # End of list

tensorboard_cb_dict={
        'name':'tensorboard',
        'log_dir':'tb_logs',
        'histogram_freq':0,
        'write_graph':True,
        'write_images':False,
        'write_steps_per_second':False,
        'update_freq':'epoch',
        'profile_batch':0,
        'embeddings_freq':0,
        'embeddings_metadata':None
    }

reduce_lr_on_plateau_cb_dict={
        'name':'reduce_lr_on_plateau',
        'monitor':'val_mean_total_loss',
        'factor':0.1,
        'patience':10,
        'verbose':1,
        'mode':'auto',
        'min_delta':1e-4,
        'cooldown':0, # Number of epochs ro wait after lr has been reduced
        'min_lr':1e-5
},

