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

                        'dataset_name_goes_here':{
                              'skiprows': None, 
                                        # desc : Number of rows to skip at the beginning of the dataset
                                        # dtype : int/None

                              'header':0, 
                                        # desc: Row number of the header
                                        # dtype : int/None

                              'path':'dataset_path_goes_here',
                                        # desc : Path to the dataset
                                        # dtype : str

                              'variables':{'first_variable_name_of_dataset1_goes_here':{'cols':[
                                                                # Columns to use in the dataset go here
                                                                ], 'preprocess': None # Preprocessing to apply to the dataset (None, std, ohe, lb, le)
                                                                },

                                            'second_variable_name_of_dataset1_goes_here':{'cols':[
                                                                # Columns to use in the dataset go here
                                                                ], 'preprocess': None # Preprocessing to apply to the dataset (None, std, ohe, lb, le)
                                                                },                                        
                                            }
                                        }

                    },

            # Dataset used for making a prediction
            'predict':{

                        'dataset_name_goes_here':{
                              'skiprows': None, # Number of rows to skip at the beginning of the dataset
                              'header':0, # Row number of the header
                              'path':'dataset_path_goes_here',
                              'variables':{'first_variable_name_of_dataset2_goes_here':{'cols':[
                                                                # Columns to use in the dataset go here
                                                                ], 'preprocess': None # Preprocessing to apply to the dataset (None, std, ohe, lb, le)
                                                                },

                                            'second_variable_name_of_dataset2_goes_here':{'cols':[
                                                                # Columns to use in the dataset go here
                                                                ], 'preprocess': None # Preprocessing to apply to the dataset (None, std, ohe, lb, le)
                                                                },                                        
                                            }
                                        }

                        }
        },

        # ... Datasets to train the second autoencoder go here ...
        {

            # This will be the data used for training. Will be split into training and validation
            'train':{

                        'dataset_name_goes_here':{
                              'skiprows': None, # Number of rows to skip at the beginning of the dataset
                              'header':0, # Row number of the header
                              'path':'dataset_path_goes_here',
                              'variables':{'first_variable_name_of_dataset3_goes_here':{'cols':[
                                                                # Columns to use in the dataset go here
                                                                ], 'preprocess': None # Preprocessing to apply to the dataset (None, std, ohe, lb, le)
                                                                },

                                            'second_variable_name_of_dataset3_goes_here':{'cols':[
                                                                # Columns to use in the dataset go here
                                                                ], 'preprocess': None # Preprocessing to apply to the dataset (None, std, ohe, lb, le)
                                                                },                                        
                                            }
                                        },

                        'latents_dataset_name_goes_here':{ 
                                'skiprows':None,
                                'header':None,
                                'path':'../runs/model_name_goes_here/submodule_outputs/predict/encoder_output.csv',
                                #'weight_samples':{'col_idx':3, 'nbins':50, 'scheme':'bin_prob'},
                                'variables':{'latents':{'cols':[
                                                                # Columns to use in the dataset go here
                                                               ], 'preprocess': None # Preprocessing to apply to the dataset (None, std, ohe, lb, le)
                                                        }
                                            }

                            },      

                    }

            }
]
