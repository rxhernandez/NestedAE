"""Neural network inputs dictionary"""

list_of_nn_params_dict=[
       {

              #'model_type':'ae1_bg_predictor_enc_l_12_l1_1em2_tanh_pred_p_0_1_100_relu_dec_15_linear_seed_0_lr_1em3_bs_10_1500_epochs_mae_mtl_k_fold_1_xavier_normal',
              'model_type':'test_ae1',

              'submodules':{

                     'encoder':{

                            'connect_to':['all_props'],
                            'num_nodes_per_layer':[12],
                            'layer_type':['linear'],
                            'layer_activation':['tanh'],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'layer_weight_reg':{'l1':0.01, 'l2':0},
                            'save_output_on_fit_end':True,
                     },

                     'bg_pred':{
                     
                           'connect_to':['encoder'],
                           'num_nodes_per_layer':[100, 1],
                           'layer_type':['linear', 'linear'],
                           'layer_activation':['relu', 'relu'],
                           'layer_kernel_init':['xavier_normal', 'xavier_normal'],
                           'layer_kernel_init_gain':[1, 1],
                           'layer_bias_init':['zeros', 'zeros'],
                           'layer_dropout':[{'type':'Dropout', 'p':0.1}, None],
                           'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'bg'},
                           'save_output_on_fit_end':True,
                     },

                     'decoder':{

                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[15],
                            'layer_type':['linear'],
                            'layer_activation':[None],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'loss':{'type':'mae',
                                    'wt':1,
                                    'target':'all_props'},
                            'save_output_on_fit_end':True,
                     }

              }

       },


       {

              #'model_type':'ae2_encoder_l_6_l1_1em2_tanh_predictor_100_p_0_1_relu_decoder_linear_seed_0_lr_1em2_bs_612_mae_concat_latents_PSC_datasets_jun14_tfl',
              #'model_type':'ae2_PCE_predictor_enc_l_2_l1_1em2_tanh_predictor_0_1_100_relu_decoder_linear_seed_0_lr_1em3_bs_100_e_1500_pass_latents_ae1_fold_0_ae2_fold_2_xavier_normal_gain_10',
              #'model_type':'Voc_pred_dataset_with_ABX_props',
              'model_type':'test_ae2',

              'submodules':{

                     'encoder':{
                         
                            'connect_to':['latents', 'etm', 'htm'],
                            'num_nodes_per_layer':[2],
                            'layer_type':['linear'],
                            'layer_activation':['tanh'],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'layer_weight_reg':{'l1':0.01, 'l2':0},
                            'save_output_on_fit_end':True,
                            'save_params':True

                     },

                     'PCE_pred':{

                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[100, 1],
                            'layer_type':['linear', 'linear'],
                            'layer_activation':['relu', 'relu'],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[10, 10],
                            'layer_bias_init':['zeros', 'zeros'],
                            'layer_dropout':[{'type':'Dropout', 'p':0.1}, None],
                            'loss':{'type':'mae',
                                    'wt':1,
                                    'target':'PCE'},
                            'save_output_on_fit_end':True,
                            'save_params':True
                     },

                     'latents_pred':{

                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[12],
                            'layer_type':['linear'],
                            'layer_activation':[None],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'loss':{'type':'mae',
                                    'wt':1,
                                    'target':'latents'}, 
                            'save_output_on_fit_end':True,
                            'save_params':True
                     },

                     'etm_pred':{

                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[7],
                            'layer_type':['linear'],
                            'layer_activation':[None],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'loss':{'type':'ce',
                                    'wt':1,
                                    'target':'etm'},
                            'save_output_on_fit_end':True,
                            'save_params':True
                     },

                     'htm_pred':{

                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[4],
                            'layer_type':['linear'],
                            'layer_activation':[None],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'loss':{'type':'ce',
                                    'wt':1,
                                    'target':'htm'},
                            'save_output_on_fit_end':True,
                            'save_params':True
                     }

              }

       }

]


