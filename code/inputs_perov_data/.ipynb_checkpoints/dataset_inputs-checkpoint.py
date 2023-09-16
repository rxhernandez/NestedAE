# Sepcify the training data inputs for each autoencoder as a nested list
# X_cols Ex : [[X_ae1], [X_ae2] ... [X_aen]]
# y_cols Ex : [[y_ae1], [y_ae2] ... [y_aen]]
# If no X_cols or y_cols for training autoencoder then specify [None]

import numpy as np

list_of_nn_datasets_dict=[
    
        {'dataset_path':'../datasets/np_viability/np_viability.xlsx',
         'train_split':0.8,
         'X_cols':np.arange(2, 26).tolist() + np.arange(27,29).tolist() + [32],
         'X_enc_cols': [2, 20, 24, 25, 32],
         'X_enc_scheme': ['ohe', 'lb', 'ohe', 'ohe', 'ohe'],
         'y_cols':None,
         'y_enc_cols':None,
         'y_enc_scheme':None,
         'standardize_X':True,
         'standardize_latents':None,
         'shuffle_data':True,
         'sample_weights':[]},
    
        {'dataset_path':'../datasets/np_viability/np_viability.xlsx',
         'train_split':0.8,
         'X_cols': np.arange(29, 32).tolist() + np.arange(33, 41).tolist(),
         'X_enc_cols':[30, 35, 36, 37, 38, 39],
         'X_enc_scheme':['ohe', 'lb', 'ohe', 'ohe', 'ohe', 'ohe'],
         'y_cols':None,
         'y_enc_cols':None,
         'y_enc_scheme':None,
         'standardize_X':True,
         'standardize_latents':True,
         'shuffle_data':True,
         'sample_weights':[]} # Column number to read sample weights
    
]
