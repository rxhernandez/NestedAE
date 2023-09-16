# If adding supervisions please specify all supervision losses before reconstruction losses

# Note 1 :
# It is necessary to specify reconstruction losses but not necessary to specify y losses.
# If no y_losses please mention None.

# Note 2 : 
# If multiple y or reconstruction losses then write as a list

list_of_nn_compile_params_dict=[

            {'y_losses':None,
            'reconst_losses':['mae'],
            'y_loss_wts':None,
            'reconst_loss_wts':[1],
            'y_metrics':None,
            'reconst_metrics':['mae'],
            'optimizer':['adam', 1e-3]},
    
            {'y_losses':None,
            'reconst_losses':['mae'],
            'y_loss_wts':None,
            'reconst_loss_wts':[1],
            'y_metrics':None,
            'reconst_metrics':['mae'],
            'optimizer':['adam', 1e-3]}

        ]
