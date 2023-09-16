"""Dataset input dictionary"""

# The way this input file is structured is as a nested dictionary
# "First" level segregates dictionaries into train and predict
# "Second" level segregates different datasets under train or predict as separate dictionaries
# "Third" level allows users to select what variables become part of the dictionary

list_of_nn_datasets_dict=[
         {

            # This will be the data used for training. Will be split into training and validation
            'train':{

                        'X':{
                              'skiprows':None,
                              'header':0,
                              'path':'../datasets/PSC_bandgaps/PSC_bandgaps_dataset.csv',
                              #'weight_samples':{'col_idx':3, 'nbins':50, 'scheme':'bin_prob'},
                              'variables':{'all_props':{'cols':[19, # A_ion_radius
                                                                23, # A_atomic_wt
                                                                24, # A_EA
                                                                25, # A_IE
                                                                28, # A_EN
                                                                31, # B_ion_radius
                                                                35, # B_atomic_wt
                                                                36, # B_EA
                                                                37, # B_IE
                                                                40, # B_EN
                                                                43, # X_ion_radius
                                                                47, # X_atomic_wt
                                                                48, # X_EA
                                                                49, # X_IE
                                                                52  # X_EN
                                                                ], 'preprocess':'std'},

                                            'bg':{'cols':[3], 'preprocess':None},                                        
                                            }
                                        }

                            },

            # Dataset used for making a prediction
            'predict':{

                        'X':{
                              'skiprows':None,
                              'header':0,
                              'path':'../datasets/PSC_efficiencies/PSC_efficiencies_dataset.csv',
                              #'weight_samples':{'col_idx':3, 'nbins':50, 'scheme':'bin_prob'},
                              'variables':{'all_props':{'cols':[20, # A_ion_radius
                                                                21, # A_atomic_wt
                                                                22, # A_EA
                                                                23, # A_IE
                                                                24, # A_EN
                                                                31, # B_ion_radius
                                                                32, # B_atomic_wt
                                                                33, # B_EA
                                                                34, # B_IE
                                                                35, # B_EN
                                                                42, # X_ion_radius
                                                                43, # X_atomic_wt
                                                                44, # X_EA
                                                                45, # X_IE
                                                                46, # X_EN
                                                                ], 'preprocess':'std'}
                                            }
                                    }
                    }
        },

        {

            # This will be the data used for training. Will be split into training and validation
            'train':{

                        'X':{
                              'skiprows':None,
                              'header':0,
                              'path':'../datasets/PSC_efficiencies/PSC_efficiencies_dataset.csv',
                              #'weight_samples':{'col_idx':3, 'nbins':50, 'scheme':'bin_prob'},
                              'variables':{'all_props':{'cols':[20, # A_ion_radius
                                                                21, # A_atomic_wt
                                                                22, # A_EA
                                                                23, # A_IE
                                                                24, # A_EN
                                                                31, # B_ion_radius
                                                                32, # B_atomic_wt
                                                                33, # B_EA
                                                                34, # B_IE
                                                                35, # B_EN
                                                                42, # X_ion_radius
                                                                43, # X_atomic_wt
                                                                44, # X_EA
                                                                45, # X_IE
                                                                46, # X_EN
                                                                ], 'preprocess':'std'},

                                            'etm':{'cols':[56], 'preprocess':'ohe'},

                                            'htm':{'cols':[57], 'preprocess':'ohe'},

                                            'PCE':{'cols':[69], 'preprocess':None},

                                            }
                            },

                        'l':{ 'skiprows':None,
                              'header':None,
                              'path':'../runs/perovskite_multiscale_dataset_3/ae1_bg_predictor_enc_l_12_l1_1em2_tanh_pred_p_0_1_100_relu_dec_15_linear_seed_0_lr_1em3_bs_10_1500_epochs_mae_mtl_k_fold_0_xavier_normal/submodule_outputs/predict/encoder_output.csv',
                              #'weight_samples':{'col_idx':3, 'nbins':50, 'scheme':'bin_prob'},
                              'variables':{'latents':{'cols':[0,
                                                              1,
                                                              2,
                                                              3,
                                                              4,
                                                              5,
                                                              6,
                                                              7,
                                                              8,
                                                              9,
                                                              10,
                                                              11], 'preprocess':'std'}
                                                              }

                            },      

                    }

        }


]
