import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
import os
import json
import pprint
import pickle
import matplotlib.pyplot as plt

# User defined libraries
from nn.vanilla_ae import VanillaAE
from nn.compile import create_optimizer_object, create_loss_list, create_metric_list

def save_to_pickle(file, file_name, save_dir):
    
    with open(save_dir + '/' + file_name, 'wb') as f:
        pickle.dump(file, f)
        
    return None
     
def set_global_random_seed(seed):
    
    print(' --> Setting global random seed {}'.format(seed))
    tf.keras.utils.set_random_seed(seed)

    return None

def read_from_pickle(file_name, save_dir):
    
    with open(save_dir + '/' + file_name, 'rb') as f:
        file = pickle.load(f)
    
    return file

def create_nn(nn_save_dir, global_seed, nn_params_dict, nn_train_params_dict, nn_compile_params_dict, X_descr, y_descr, run_eagerly):

    if nn_params_dict['model_name'].split('_')[0]=='vanillaAE':
        if y_descr == None:
            # Initialize vanilla ae
            nn = VanillaAE(global_seed, nn_params_dict, nn_train_params_dict, nn_compile_params_dict, X_descr[0], X_descr[1])
        else:
            # Initialize supervised vanilla ae
            nn = VanillaAE(global_seed, nn_params_dict, nn_train_params_dict, nn_compile_params_dict, 
                           X_descr[0], X_descr[1],
                           y_descr[0], y_descr[1])

        loss_wts = []
        if 'y_loss_wts' in nn_compile_params_dict.keys() and \
            nn_compile_params_dict['y_loss_wts'] != None:
            loss_wts = loss_wts + nn_compile_params_dict['y_loss_wts']

        if 'X_loss_wts' in nn_compile_params_dict.keys() and \
            nn_compile_params_dict['X_loss_wts'] != None:
            loss_wts = loss_wts + nn_compile_params_dict['X_loss_wts']
            
        nn.compile(optimizer=create_optimizer_object(nn_compile_params_dict),
                   loss=create_loss_list(nn_compile_params_dict, True),
                   metrics=create_metric_list(nn_compile_params_dict, True),
                   loss_weights=loss_wts,
                   run_eagerly=run_eagerly)

        # Call the model on some inputs 
        #X = []
        #for dim in X_descr[1]:
        #    #X.append(np.zeros(shape=(dim,)))
        #    X.append(tf.keras.Input(shape=(dim,)))

        if run_eagerly == 'true' or run_eagerly == 'True':

            X = []
            for dim in X_descr[1]:
                X.append(tf.keras.Input(shape=(dim,)))


            # Save the model plot
            tf.keras.utils.plot_model(Model(inputs=X, outputs=nn.call(X)),
                                      to_file=nn_save_dir + '/' + 'nn_plot.png',
                                      show_shapes=True, 
                                      show_dtype=False,
                                      show_layer_names=True,
                                      rankdir="TB", 
                                      dpi=96,
                                      show_layer_activations=True) 

        #X = np.asarray(X).reshape(None, len())
        
    elif nn_params_dict['model_name'].split('_')[0]=='vanillaNN':
        #list_of_model_objects.append(VanillaAE(i))
        #nn = VanillaNN()
        raise NotImplementedError
    else:
        #list_of_model_objects.append(VanillaAE(i))
        raise NotImplementedError
    
    return nn


# Test cell 
#obj = compile_nn_object_list[0]
#print(dir(obj))
#print(obj.__class__)
#print(obj.__dict__)
