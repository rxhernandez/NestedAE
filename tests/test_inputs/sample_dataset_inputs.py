import pytest

# Complete set of supported options for nn_datasets_dict
@pytest.fixture
def sample_nn_datasets_dict():
    return {'train':{ 
                # Required 
                # Desc : Dictionary of training datasets. 
                #        The 'key' for this dictionary is called the dataset_type
                #        which is passed as an argument in the go.sh script
                # Supported input type : dict

                'sample_train_dataset':{
                    # Required
                    # Desc : Dictionary containing dataset information
                    # Supported input type : dict

                    'skiprows':None, 
                        # Optional (default is None)
                        # Desc : Number of rows to skip at the beginning of the csv/excel file
                        # Supported input type : int/None

                    'header':0, 
                        # Optional (default is 0)
                        # Desc : Row number of the header
                        # Supported input type : int/None

                    'path':'tests/test_datasets/sample_train_dataset.csv', 
                        # Required
                        # Desc : Path to the dataset 
                        # Currently supported dataset types are csv or excel.
                        # Supported input type : str

                    'descriptors':{ 
                        'desc1':{'cols':[0], 'preprocess':None},
                            # Required
                            # Desc : descriptor dictionary. Specify the columns of
                            #       the csv/excel file to use for the descriptor and
                            #       how to preprocess it. 
                            # Currently supported preprocessing types are One Hot Encoding ('ohe') and standardization ('std')
                            # Supported input type : dict

                        'desc2':{'cols':[1], 'preprocess':'std'},
                        'desc3':{'cols':[2], 'preprocess':'ohe'}},
                        # Required
                        # Desc : Dictionary of descriptors
                        # Supported input type : dict
                }},

        'test':{'sample_test_dataset':{
                    'skiprows':None,
                    'header':0,
                    'path':'tests/test_datasets/sample_test_dataset.csv',
                    'descriptors':{
                        'desc1':{'cols':[0]},
                        'desc2':{'cols':[1]},
                        'desc3':{'cols':[2]}},
                    'load_preprocessor':True
                        # Optional (default is False)
                        # Desc : Load preprocessor. 
                        #        This is necessary when you are testing on new data 
                        # Supported input type : bool
                }
            }           
}