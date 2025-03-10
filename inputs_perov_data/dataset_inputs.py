"""Dataset input dictionary"""

# The way this input file is structured is as a nested dictionary
# "First" level segregates dictionaries into train and predict
# "Second" level segregates different datasets under train or predict as separate dictionaries
# "Third" level allows users to select what variables become part of the dictionary

list_of_nn_datasets_dict=[

        # ... Datasets to train the first autoencoder go here ...
        {

            # This will be the data used for training. Will be split into training and validation
            'train':{

                        'X':{
                              'skiprows': None, 
                                        # desc : Number of rows to skip at the beginning of the dataset
                                        # dtype : int/None
                              'header':0, 
                                        # desc: Row number of the header
                                        # dtype : int/None
                              'path':'https://raw.githubusercontent.com/mannodiarun/perovs_mfml_ga/refs/heads/run_ml/Expt_data.csv',
                                        # desc : Path to the dataset
                                        # dtype : str
                        #       'descriptors':{'K':{'cols':[5], 'preprocess': None},
                        #                      'Rb':{'cols':[6], 'preprocess': None},
                        #                      'Cs':{'cols':[7], 'preprocess': None},
                        #                      'MA':{'cols':[8], 'preprocess': None},
                        #                      'FA':{'cols':[9], 'preprocess': None},
                        #                      'Ca':{'cols':[10], 'preprocess': None},
                        #                      'Sr':{'cols':[11], 'preprocess': None},
                        #                      'Ba':{'cols':[12], 'preprocess': None},
                        #                      'Ge':{'cols':[13], 'preprocess': None},
                        #                      'Sb':{'cols':[14], 'preprocess': None},
                        #                      'Pb':{'cols':[15], 'preprocess': None},
                        #                      'Cl':{'cols':[16], 'preprocess': None},
                        #                      'Br':{'cols':[17], 'preprocess': None},
                        #                      'I':{'cols':[18], 'preprocess': None},
                        #                      'bg':{'cols':[3], 'preprocess':None}},
                        'descriptors':{
                                'A_comp':{'cols':[5, 6, 7, 8, 9], 'preprocess': None},
                                'B_comp':{'cols':[10, 11, 12, 13, 14, 15], 'preprocess': None},
                                'X_comp':{'cols':[16, 17, 18], 'preprocess': None},
                                'bg':{'cols':[3], 'preprocess':None}
                        },
                              'load_preprocessor':False
                              }
                    }
        }
]
