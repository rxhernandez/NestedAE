list_of_nn_params_dict=[

       {
              'model_type':'nestedAE_AE1_fold4',

              'submodules':{

                     'encoder':{
                            'connect_to':['x1tox8'],
                            'hidden_dim':{'values':[25, 50]},
                            'hidden_layers':{'values':[1, 2]},
                            'output_dim':{'values':[2, 4, 6, 8]},
                            'layer_type':{'value':'linear'},
                            'layer_activation':{'values':[None, 'relu', 'tanh']},
                            'layer_kernel_init':{'value':'xavier_normal'},
                            'layer_kernel_init_gain':{'value':1},
                            'layer_bias_init':{'value':'zeros'},
                            'layer_weight_reg_l1':{'value':0},
                            'layer_weight_reg_l2':{'value':0.001}
                     },

                     'decoder':{
                            'connect_to':['encoder'],
                            'output_activation':{'value':None},
                            'output_dim':{'value':8},
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'x1tox8'},
                     }

              }

       },


       {

              'model_type':'nestedAE_AE2_targetf6_fold4',

              'submodules':{

                     'encoder':{
                            'connect_to':['f1tof4_w_latents'],
                            'hidden_dim':{'values':[25, 50]},
                            'hidden_layers':{'values':[1, 2]},
                            'output_dim':{'values':[6, 8, 10, 12]},
                            'layer_type':{'value':'linear'},
                            'layer_activation':{'values':[None, 'relu', 'tanh']},
                            'layer_kernel_init':{'value':'xavier_normal'},
                            'layer_kernel_init_gain':{'value':1},
                            'layer_bias_init':{'value':'zeros'},
                            'layer_weight_reg_l1':{'value':0},
                            'layer_weight_reg_l2':{'value':0.001}
                     },

                     'predictor':{
                            'connect_to':['encoder'],
                            'output_activation':{'value':None},
                            'output_dim':{'value':1},
                            'loss':{'type':'mae',
                                    'wt':1,
                                    'target':'f6'}
                     },

                     'decoder':{
                            'connect_to':['encoder'],
                            'output_activation':{'value':None},
                            'output_dim':{'value':12},
                            'loss':{'type':'mae',
                                    'wt':1,
                                    'target':'f1tof4_w_latents'}
                     }

              }

       }

]

# Code to add the latents from AE1 to the synthetic_data_randomSamples_200.csv
# from nn.vanilla_ae import VanillaAE
# import yaml 

# AE_dir = '../runs/results_for_RL_paper/nestedAE_AE1_fold0/tune_nn_params/brisk-sweep-4_fold_0'

# nn_save_dir = yaml.safe_load(open(AE_dir + '/files/config.yaml'))['nn_save_dir']['value']
# nn_params_dict = yaml.safe_load(open(AE_dir + '/files/config.yaml'))['nn_params_dict']['value']
# nn_train_params_dict = yaml.safe_load(open(AE_dir + '/files/config.yaml'))['nn_train_params_dict']['value']
# nn_datasets_dict = yaml.safe_load(open(AE_dir + '/files/config.yaml'))['nn_datasets_dict']['value']

# new_AE = VanillaAE(nn_save_dir, nn_params_dict, nn_train_params_dict, nn_datasets_dict)
# os.mkdir('temp')
# shutil.copy(AE_dir + '/checkpoints/last.ckpt', 'temp')
# loaded_AE = new_AE.load_from_checkpoint('temp/last.ckpt')
# shutil.rmtree('temp')

# submodule_outputs = loaded_AE(loaded_AE.all_samples)

# synthetic_dataset = pd.read_csv('../datasets/synthetic_dataset/random_sampling/synthetic_data_randomSamples_200.csv')
# # Add the latents as new columns to the synthetic dataset
# synthetic_dataset['AE1_latent_0'] = submodule_outputs['encoder'][:, 0].detach().numpy()
# synthetic_dataset['AE1_latent_1'] = submodule_outputs['encoder'][:, 1].detach().numpy()
# synthetic_dataset['AE1_latent_2'] = submodule_outputs['encoder'][:, 2].detach().numpy()
# synthetic_dataset['AE1_latent_3'] = submodule_outputs['encoder'][:, 3].detach().numpy()
# synthetic_dataset['AE1_latent_4'] = submodule_outputs['encoder'][:, 4].detach().numpy()
# synthetic_dataset['AE1_latent_5'] = submodule_outputs['encoder'][:, 5].detach().numpy()
# synthetic_dataset['AE1_latent_6'] = submodule_outputs['encoder'][:, 6].detach().numpy()
# synthetic_dataset['AE1_latent_7'] = submodule_outputs['encoder'][:, 7].detach().numpy()

# # Save the dataset with the latents as new columns
# synthetic_dataset.to_csv('../datasets/synthetic_dataset/random_sampling/synthetic_data_randomSamples_200_with_AE1_latents.csv', index=False)