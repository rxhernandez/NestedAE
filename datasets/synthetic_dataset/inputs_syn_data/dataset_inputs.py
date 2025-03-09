 ################################################################################################
# Synthetic Database dictionary
################################################################################################

#'variables':{'f1tof4':{'cols':[0,1,2,3], 'preprocess':'std'}
list_of_nn_datasets_dict=[

    {

        'train':{
            
                    'X1':{
                            'skiprows':1, # Skips the header row
                            'header':None,
                            'path':'../datasets/synthetic_dataset/random_sampling/synthetic_data_randomSamples_200.csv',
                            'variables':{'x1tox8':{'cols':[0,1,2,3,4,5,6,7], 'preprocess':'std'}},
                            'load_preprocessor':False
                        },
        }

    },

    {

        'train':{
            
                    'X2':{
                            'skiprows':1, # Skips the header row
                            'header':None,
                            'path':'../datasets/synthetic_dataset/random_sampling/synthetic_data_randomSamples_200_with_AE1_latents.csv',
                            'variables':{'f1tof4_w_latents':{'cols':[8,9,10,11,14,15,16,17,18,19,20,21], 'preprocess':'std'},
                                         'f5':{'cols':[12], 'preprocess':None},
                                         'f6':{'cols':[13], 'preprocess':None}},
                            'load_preprocessor':False
                        },
        }

    },



    # {

    #     'train':{

    #                 'X2':{
    #                         'skiprows':1,
    #                         'header':None,
    #                         'path':'../runs/noisyF_l2-5_lr_1em4_l1_1em3_l2_0_tanh_200pts/synthetic_data_FuncAddNoise_200.csv',
    #                         'variables':{'f5tof13andae1l':{'cols':[9,10,11,12,13,14,15,16,17,18,19], 'preprocess':'std'}}
    #                     }

    #     }
    # }
]
