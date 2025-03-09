list_of_nn_datasets_dict=[
    
        {'dataset_path':'../datasets/synthetic_dataset/synthetic_data.csv',
         'test_split':0.2,
         'X_cols': [[0,1,2,3]],
         'X_preprocess_scheme': ['std'],
         'y_cols':None,
         'y_preprocess_scheme':None,
         'add_latents':None,
         'latents_preprocess_scheme':None, 
         'shuffle_data':True,
         'sample_weights':[]},
    
        {'dataset_path':'../datasets/synthetic_dataset/synthetic_data.csv',
         'test_split':0.2,
         'X_cols':[[4,5,6,7,8,9,10,11,12]],
         'X_preprocess_scheme':['std'],
         'y_cols':None,
         'y_preprocess_scheme':None,
         'add_latents':'vanilla_ae_1',
         'latents_preprocess_scheme':'std',
         'shuffle_data':True,
         'sample_weights':[]}#TODO(nthota2):Mention column number to read sample weights from
    
]
