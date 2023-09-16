" VanillaAE class "

# Pytorch libraries
import torch
from torch import nn
from torchmetrics import MeanAbsolutePercentageError, Accuracy, MeanSquaredError
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import pprint
import sys
import os

# User defined libraries
from utils.nn_utils import check_dict_key_exists, get_module_input_dim
from utils.nn_utils import set_layer_activation, set_layer_dropout, set_layer_init, create_loss_object, create_optimizer_object, create_scheduler_object

class VanillaAE(pl.LightningModule):
    """VanillaAE class"""
    def __init__(self,
                 nn_save_dir,
                 nn_params_dict,
                 nn_train_params_dict,
                 nn_datasets_dict):

        super(VanillaAE, self).__init__()

        # Save the parameters passed into __init__
        self.save_hyperparameters()
        self.automatic_optimization = True
        self.model_type = nn_params_dict['model_type']
        self.model_dir = nn_save_dir
        self.global_seed = nn_train_params_dict['global_seed']
        self.nn_params_dict = nn_params_dict
        self.nn_train_params_dict = nn_train_params_dict
        self.nn_datasets_dict = nn_datasets_dict
        self.datasets = torch.load(nn_save_dir+'/datasets/dataset.pt')
        self.all_samples = self.datasets[:]
        self.example_input = self.datasets[0]
        self.submodule_dicts = nn_params_dict['submodules']
        self.submodule_losses = None
        # Create a dictionary to store all submodules
        self.submodules = torch.nn.ModuleDict()
        self.trace_model = False
        # Init variables to store batch and epoch loss results
        self.train_loss_values = None
        self.val_loss_values = None
        self.train_losses_batch = None
        self.val_losses_batch = None
        self.train_losses_epoch = None
        self.val_losses_epoch = None
        # Save outputs on epoch to pickle
        self.save_outputs_on_epoch = {}

        # Outer loop iterates over the submodules
        for submodule_name, submodule_dict in \
            zip(self.submodule_dicts.keys(), self.submodule_dicts.values()):

            # If building a variational autoencoder then 'z' is a special submodule
            if submodule_name == 'z':
                layer_list = None
            else:
                layer_list = torch.nn.ModuleList()
                num_nodes_per_layer = submodule_dict['num_nodes_per_layer']

                layer_kernel_init_list = []
                layer_kernel_init_list.extend(submodule_dict['layer_kernel_init'])

                layer_kernel_init_gain_list = []
                layer_kernel_init_gain_list.extend(submodule_dict['layer_kernel_init_gain'])

                layer_bias_init_list = []
                layer_bias_init_list.extend(submodule_dict['layer_bias_init'])

                layer_type_list = []
                layer_type_list.extend(submodule_dict['layer_type'])

                layer_activation_list = []
                layer_activation_list.extend(submodule_dict['layer_activation'])

                if check_dict_key_exists('layer_dropout', submodule_dict):
                    layer_dropout_list = []
                    if isinstance(submodule_dict['layer_dropout'], list):
                        layer_dropout_list.extend(submodule_dict['layer_dropout'])
                    else:
                        layer_dropout_list.append(submodule_dict['layer_dropout'])
                else:
                    layer_dropout_list = None

                # Inner loop iterates over the layers in each submodule
                for j, num_nodes in enumerate(num_nodes_per_layer):

                    # The first layer receives inputs from a submodule or dataset or both.
                    if j == 0:
                        # Calculate the input dimensions to first layer based on all inputs connected to that layer
                        input_dim = get_module_input_dim(submodule_dict['connect_to'], \
                                                         self.nn_params_dict, \
                                                         self.datasets.variable_shapes)
                    else:
                        # Get the input dimensions of every subsequent layer from out dims of previous layer 
                        input_dim = num_nodes_per_layer[j - 1]

                    # Future Support
                    if layer_type_list[j] == 'linear':
                        layer_list.append(nn.Linear(in_features=input_dim,
                                                    out_features=num_nodes,
                                                    bias=True))
                    else:
                        raise ValueError(' --> Unknown layer type.')

                    # Add any activations if specified
                    if (layer_activation_list[j] is not None) and (layer_activation_list[j] != 'linear'):
                        activation = layer_activation_list[j]
                        layer_list.append(set_layer_activation(activation))

                    # Add dropout after each layer if specified
                    if layer_dropout_list is not None:
                        if layer_dropout_list[j] is not None:
                            dropout_type = layer_dropout_list[j]['type']
                            p = layer_dropout_list[j]['p']
                            layer_list.append(set_layer_dropout(dropout_type, p))

                # Initialize weights for all layers
                layer_list = set_layer_init(layer_list,\
                                            layer_activation_list,\
                                            layer_kernel_init_list,\
                                            layer_kernel_init_gain_list,\
                                            layer_bias_init_list)

            # Check to see if a submodule has to be loaded
            if check_dict_key_exists('load_params', submodule_dict):
                path = submodule_dict['load_params']
                try:
                    submodule_params = torch.load(path)
                    layer_list.load_state_dict(submodule_params['state_dict'])
                    print(f' --> Loaded submodule {submodule_name}.')
                except OSError as err:
                    raise FileNotFoundError(f' --> Could not load {submodule_name}.') from err

            # Finally add to submodule list
            self.submodules.update({submodule_name:layer_list})

            print('\n')
            print(f' --> Submodule {submodule_name} layers :')
            print(layer_list)

    def forward(self, module_inputs):
        """Forward pass through the model."""

        # Stores all submodule outputs
        submodule_outputs = {}

        module_input_ids = list(module_inputs.keys())

        # Outer loop iterates over the submodules
        for submodule_name, submodule in \
             zip(self.submodules.keys(), self.submodules.values()):

            # Get the input ids that are connected to the submodule
            submodule_input_ids = self.submodule_dicts[submodule_name]['connect_to']
            # Ensure that the input ids are in list form
            submodule_input_ids = submodule_input_ids \
                                    if isinstance(submodule_input_ids, list) \
                                    else list(submodule_input_ids)

            # Case : input sampled from a probability distribution
            if submodule_name == 'z':
                if self.submodule_dicts['z']['sample_from'] == 'normal':
                    # Required submoduel keys
                    mu, logvar = submodule_outputs['mu'], submodule_outputs['logvar']
                    inp = [mu, logvar]
                    std = logvar.exp().mul(0.5)   
                    eps = torch.randn_like(std)
                    output = eps*std + mu
                else:
                    raise ValueError('Reparameterization from chosen distribution not defined.\
                                    Please add Reparameterization scheme in forward().')
                
            # Case : input comes from a dataset or another submodule
            else:
                inp = []
                for submodule_input_id in submodule_input_ids:
                    if submodule_input_id in module_input_ids:
                        inp.append(module_inputs[submodule_input_id])
                    else: # Then input is output from previous module
                        inp.append(submodule_outputs[submodule_input_id])

                # Convert list of tensors to a single tensor
                inp = torch.concatenate(inp, dim=-1)

                for j, layer in enumerate(submodule):

                    if j == 0:
                        output = layer(inp)

                    else:
                        output = layer(output)

            submodule_outputs[submodule_name] = output

            if self.trace_model:
                print('\n')
                print(' ---------------------------------- ')
                print(f'module_name:{submodule_name}')
                print(f'input id:{submodule_input_ids}')
                print('input to submodule :')
                print(inp)
                print(f'output id:{submodule_name}')
                print('output from submodule :')
                print(output)
                print('Submodule output dictionary :')
                pp = pprint.PrettyPrinter()
                pp.pprint(submodule_outputs)
                print(' ---------------------------------- ')
                print('\n')

        return submodule_outputs

    # <-- TESTING -->
    # Uncomment for using custom pytorch training loops
    #def backward(self, loss, optimizer):
    #    loss.backward()

    def compile(self):
        """Compile the model."""
        losses = {}
        for submodule_name, submodule_dict in \
            zip(self.submodule_dicts.keys(), self.submodule_dicts.values()):

            if 'loss' in list(submodule_dict.keys()):
                losses[submodule_name] = create_loss_object(submodule_dict['loss']['type'])

        self.submodule_losses = losses

    def configure_optimizers(self):
        """Configure the optimizers."""
        optimizer = create_optimizer_object(self.submodules, self.nn_train_params_dict)

        if 'scheduler' in list(self.nn_train_params_dict.keys()):
            scheduler = create_scheduler_object(optimizer, self.nn_train_params_dict)
            return {'optimizer':optimizer, 'lr_scheduler':scheduler}

        return {'optimizer':optimizer}

    def training_step(self, batch, batch_idx):
        """Training step."""
        # <-- TESTING -->
        #torch.set_grad_enabled(True)
        # Call the  optimizers
        #opt = self.optimizers()
        #opt.zero_grad()

        # Squeeze all the tensors
        #for dataset in batch.keys():
        #    for variable in batch[dataset].keys():
        #        batch[dataset][variable] = torch.squeeze(batch[dataset][variable])

        # Pass data into model
        submodule_outputs = self(batch)

        train_loss_values = {}

        # Init total loss to 0
        total_train_loss = torch.tensor(0, device=self.device, dtype=torch.float32)

        # Calculate reconstruction mape for each feature ?
        calc_mape_for_each_feat = False

        # Prediction losses
        for submodule_name, submodule_loss in \
            zip(self.submodule_losses.keys(), self.submodule_losses.values()):

            # Output from submodule
            output = submodule_outputs[submodule_name]

            variable_name = self.submodule_dicts[submodule_name]['loss']['target']
            target = batch[variable_name]

            loss_type = self.submodule_dicts[submodule_name]['loss']['type']
            loss_wt = self.submodule_dicts[submodule_name]['loss']['wt']
            loss_wt = torch.tensor(loss_wt, device=self.device, dtype=torch.float32, requires_grad=False)

            if 'sample_wts' in list(batch.keys()):
                sample_wt = batch['sample_wts']
            else:
                sample_wt = torch.tensor(1, device=self.device, dtype=torch.float32, requires_grad=False)
            
            pred_loss = submodule_loss(output, target).multiply(loss_wt).multiply(sample_wt)

            total_train_loss = total_train_loss.add(pred_loss)

            # NOTE : 
            # -> Best result is 0. Bad predictions can lead to arbitrarily large values.
            # -> This occurs when the target is close to 0. MAPE returns a large number instead of inf
            if loss_type == 'mae' or loss_type == 'huber':
                # metric = MeanAbsolutePercentageError()
                # metric_name = 'mape'
                # metric_train = metric(output, target)
                metric = MeanSquaredError(squared=False)
                metric_name = 'rmse'
                metric_train = metric(output, target)
            elif loss_type == 'mse':
                metric = MeanSquaredError(squared=False)
                metric_name = 'rmse'
                metric_train = metric(output, target)
            elif loss_type == 'ce':
                num_classes = target.size(1)
                metric = Accuracy(task="multilabel", num_labels=num_classes, average='macro')
                metric_name = 'accuracy'
                #output = torch.nn.LogSoftmax(dim=-1)(output)
                metric_train = metric(output, target)
            elif loss_type == 'bcewithlogits':
                metric = Accuracy(task="binary", threshold=0.5)
                metric_name = 'binary_accuracy'
                #output = torch.nn.LogSoftmax(dim=-1)(output)
                metric_train = metric(output, target)

            # The weighted Pred loss
            train_loss_values['train_'+ variable_name + '_' + loss_type] = pred_loss.item()
            train_loss_values['train_'+ variable_name + '_' + metric_name] = metric_train.item()

            # Calculate the mape for each input in the inputs vector
            if calc_mape_for_each_feat:
                mape = MeanAbsolutePercentageError()
                for i in range(0, output.size(1)):
                    train_loss_values['train_mape_'+str(i)] = mape(output[:, i], target[:, i]).item()

        # <-- USER START -->
        # Can define custom losses here
        # Make sure to log your loss after the total loss is logged so
        # that plot_utils.py -u metrics can create the appropriate train and val prediction loss plots

        # Access the outputs of any hidden layer by : output = submodule_outputs['layer_name']

        compute_kld_loss = False

        # KLD Loss for variational autoencoder
        if compute_kld_loss:

            if self.submodule_dicts['z']['sample_from'] == 'normal':
                mu = submodule_outputs['mu']
                logvar = submodule_outputs['logvar']

                # torch.sum() calculates the kld_loss for a single sample e
                # torch.mean() calculated the mean kld_loss over mini batch
                kld_loss = torch.mean(-0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

            # Beta VAE : https://openreview.net/forum?id=Sy2fzU9gl
            beta = torch.tensor(1, device=self.device, dtype=torch.float32, requires_grad=False)

            if 'sample_wts' in list(batch.keys()):
                sample_wt = batch['sample_wts']['wts']
                sample_wt = torch.tensor(sample_wt, device=self.device, dtype=torch.float32, requires_grad=False)
            else:
                sample_wt = torch.tensor(1, device=self.device, dtype=torch.float32, requires_grad=False)

            kld_loss = kld_loss.multiply(beta).multiply(sample_wt)

            total_train_loss = total_train_loss.add(kld_loss)

        # <-- USER END -->

        # Init Regularization losses
        l1_weight_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        l2_weight_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        l1_bias_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        l2_bias_loss = torch.tensor(0, device=self.device, dtype=torch.float32)

        # Regularization losses
        for submodule_name, submodule_dict in \
                zip(self.submodule_dicts.keys(), self.submodule_dicts.values()):
            if 'layer_weight_reg' in list(submodule_dict.keys()):
                if 'l1' in list(submodule_dict['layer_weight_reg'].keys()):
                    lambda_l1 = submodule_dict['layer_weight_reg']['l1']
                    lambda_l1 = torch.tensor(lambda_l1, device=self.device, dtype=torch.float32, requires_grad=False)
                    for name, params in self.submodules[submodule_name].named_parameters():
                        if name.endswith('.weight'):
                            w = params.view(-1) 
                            l1_weight_loss = l1_weight_loss.add(w.abs().sum().multiply(lambda_l1))
                if 'l2' in list(submodule_dict['layer_weight_reg'].keys()):
                    lambda_l2 = submodule_dict['layer_weight_reg']['l2']
                    lambda_l2 = torch.tensor(lambda_l2, device=self.device, dtype=torch.float32, requires_grad=False)
                    for name, params in self.submodules[submodule_name].named_parameters():
                        if name.endswith('.weight'):
                            # Params view is important here since weights is a 2D tensor which we unwrap to a 1D tensor
                            # params.data :- returns the weight data. No reshape
                            # params.view :- returns the weight data. With reshape
                            w = params.view(-1)
                            l2_weight_loss = l2_weight_loss.add(w.pow(2).sum().multiply(lambda_l2))
            # Future Support
            if 'layer_bias_reg' in list(submodule_dict.keys()):
                pass

        # Add in regularization losses
        total_train_loss = total_train_loss.add(l1_weight_loss).add(l2_weight_loss).add(l1_bias_loss).add(l2_bias_loss)

        # The weighted loss
        train_loss_values['total_train_loss'] = total_train_loss.item()

        # KLD loss
        if compute_kld_loss:
            train_loss_values['kld_loss'] = kld_loss.item()

        train_loss_values['l1_weight_loss'] = l1_weight_loss.item()
        train_loss_values['l2_weight_loss'] = l2_weight_loss.item()
        train_loss_values['l1_bias_loss'] = l1_bias_loss.item()
        train_loss_values['l2_bias_loss'] = l2_bias_loss.item()
        
        #self.log('total_train_loss', train_loss_values['total_train_loss'], on_epoch=True, prog_bar=True)

        self.train_loss_values = train_loss_values
        self.log_dict(train_loss_values, on_step=False,
                        on_epoch=True, logger=True, prog_bar=False,
                        batch_size=self.nn_train_params_dict['batch_size'])

        return total_train_loss

        # <-- USER START -->

        # Manually backward total loss
        #self.manual_backward(total_train_loss)
        #opt.step()
        #if self.lr_schedulers() != None:
        #    sch = self.lr_schedulers()
        #    sch.step()
        #torch.set_grad_enabled(False)

        # <-- USER END -->

    def validation_step(self, batch, batch_idx):
        """Validation step."""

        #torch.set_grad_enabled(False)

        # Squeeze all the tensors
        #for dataset in batch.keys():
        #    for variable in batch[dataset].keys():
        #        batch[dataset][variable] = torch.squeeze(batch[dataset][variable])

        # Pass data into model
        submodule_outputs = self(batch)

        # Init total loss to 0
        # Change this to get device from user requested device
        total_val_loss = torch.tensor(0, device=self.device, dtype=torch.float32)

        val_loss_values = {}

        # Calculate reconstruction mape for each input
        calc_mape_for_each_feat = False

        # Prediction losses
        for submodule_name, submodule_loss in \
                zip(self.submodule_losses.keys(), self.submodule_losses.values()):

            # Output from submodule
            output = submodule_outputs[submodule_name]

            variable_name = self.submodule_dicts[submodule_name]['loss']['target']
            target = batch[variable_name]

            loss_type = self.submodule_dicts[submodule_name]['loss']['type']
            loss_wt = self.submodule_dicts[submodule_name]['loss']['wt']
            loss_wt = torch.tensor(loss_wt, device=self.device, dtype=torch.float32, requires_grad=False)

            if 'sample_wts' in list(batch.keys()):
                sample_wt = batch['sample_wts']['wts']
                sample_wt = torch.tensor(sample_wt, device=self.device, dtype=torch.float32)
            else:
                sample_wt = torch.tensor(1, device=self.device, dtype=torch.float32)

            pred_loss = submodule_loss(output, target).multiply(loss_wt).multiply(sample_wt)

            total_val_loss = total_val_loss.add(pred_loss)

            #l1_loss = nn.L1Loss()
            #l1_pred_loss = l1_loss(output, target).multiply(loss_wt).multiply(sample_wt)

            if loss_type == 'mae' or loss_type == 'huber':
                metric = MeanSquaredError(squared=False)
                metric_name = 'rmse'
                metric_val = metric(output, target)
                # metric = MeanAbsolutePercentageError()
                # metric_name = 'mape'
                # metric_val = metric(output, target)
            elif loss_type == 'mse':
                metric = MeanSquaredError(squared=False)
                metric_name = 'rmse'
                metric_val = metric(output, target)
            elif loss_type == 'ce':
                num_classes = target.size(1)
                metric = Accuracy(task="multilabel", num_labels=num_classes, average='macro')
                metric_name = 'accuracy'
                metric_val = metric(output, target)
            elif loss_type == 'bcewithlogits':
                metric = Accuracy(task="binary", threshold=0.5)
                metric_name = 'binary_accuracy'
                #output = torch.nn.LogSoftmax(dim=-1)(output)
                metric_val = metric(output, target)

            # The weighted loss
            val_loss_values['val_'+ variable_name + '_' + loss_type] = pred_loss.item()
            val_loss_values['val_'+ variable_name + '_' + metric_name] = metric_val.item()
            #self.log('val_'+ variable + '_' + loss_type, loss_value.item(), on_step=False, on_epoch=True, prog_bar=True)

            if calc_mape_for_each_feat:
                mape = MeanAbsolutePercentageError()
                for i in range(0, output.size(1)):
                    val_loss_values['val_mape_'+str(i)] = mape(output[:, i], target[:, i]).item()

        val_loss_values['total_val_loss'] = total_val_loss.item()
        #self.log('total_val_loss', val_loss_values['total_val_loss'], on_epoch=True, prog_bar=True)

        self.val_loss_values = val_loss_values

        self.log_dict(val_loss_values, 
                        on_step=False,
                        on_epoch=True, 
                        logger=True, 
                        prog_bar=False,
                        batch_size=self.nn_train_params_dict['batch_size'])

        #torch.set_grad_enabled(True)

    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        return self(batch)

    def test_step(self, batch, batch_idx):
        """Test step."""

    ################################################################################################
    # Pytorch Lightning Model Hooks go here
    ################################################################################################
    
    def on_fit_start(self):
        """Called when fit begins."""

        # Create csv logs dir if it does not exist
        logs_dir = self.model_dir + '/logs'
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)

        csv_logs_dir = logs_dir + '/csv_logs'
        if not os.path.exists(csv_logs_dir):
            os.mkdir(csv_logs_dir)

        # Show example of input to model
        print(' --> Example Input : ')
        print(self.example_input)
        print('\n')

        print('--> Model Trace : ')
        self.trace_model = True
        # Check model hierarchy 
        module_out = self(self.example_input)
        self.trace_model = False

        sys.stdout.close()
        sys.stdout = sys.__stdout__

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """ Called at the end of training batch."""

        if isinstance(self.train_losses_batch, pd.core.frame.DataFrame) is False:
            self.train_losses_batch = pd.DataFrame(self.train_loss_values, index=[batch_idx])
        else:
            self.train_losses_batch = pd.concat([self.train_losses_batch, pd.DataFrame(self.train_loss_values, index=[batch_idx])], axis=0)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Called at the end of validation batch."""

        if isinstance(self.val_losses_batch, pd.core.frame.DataFrame) is False:
            self.val_losses_batch = pd.DataFrame(self.val_loss_values, index=[batch_idx])
        else:
            self.val_losses_batch = pd.concat([self.val_losses_batch, pd.DataFrame(self.val_loss_values, index=[batch_idx])], axis=0)

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""

        epoch_results = self.train_losses_batch.apply(np.mean, axis=0, raw=True, result_type='expand').to_frame().transpose()
        epoch_results.rename(index={0:self.current_epoch}, inplace=True)

        # Reduce the values in each column
        if isinstance(self.train_losses_epoch, pd.core.frame.DataFrame) is False:
            self.train_losses_epoch = epoch_results
            self.train_losses_batch = None
        else:
            self.train_losses_epoch = pd.concat([self.train_losses_epoch, epoch_results], axis=0)
            self.train_losses_batch = None

        # Save dataframe to csv
        csv_logs_dir = self.model_dir + '/logs/csv_logs'
        self.train_losses_epoch.to_csv(csv_logs_dir+'/'+'train_logs.csv', sep=',')

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Reduce the values in each column
        epoch_results = self.val_losses_batch.apply(np.mean, axis=0, raw=True, \
                                                    result_type='expand').to_frame().transpose()
        epoch_results.rename(index={0:self.current_epoch}, inplace=True)
        if not isinstance(self.val_losses_epoch, pd.core.frame.DataFrame):
            self.val_losses_epoch = epoch_results
            self.val_losses_batch = None
        else:
            self.val_losses_epoch = pd.concat([self.val_losses_epoch, epoch_results], axis=0)
            self.val_losses_batch = None

        # Save dataframe to csv
        csv_logs_dir = self.model_dir + '/logs/csv_logs'
        self.val_losses_epoch.to_csv(csv_logs_dir+'/'+'val_logs.csv', sep=',')

        # At the end of the validation epoch call the model on all dataset inputs
        submodule_outputs = self(self.all_samples)

        # Store the submodule outputs
        for submodule_name, submodule_dict in \
            zip(self.submodule_dicts.keys(), self.submodule_dicts.values()):

            if check_dict_key_exists('save_output_on_epoch_end', submodule_dict):

                submodule_output = submodule_outputs[submodule_name]

                if self.current_epoch == 0:
                    self.save_outputs_on_epoch[submodule_name] = [submodule_output]
                else:
                    self.save_outputs_on_epoch[submodule_name].append(submodule_output)

    def on_fit_end(self):
        """Called at the end of fit() to do things such as logging, saving etc."""

        # Create a submodule outputs dir if it does not exist
        submodule_outputs_dir = self.model_dir + '/submodule_outputs'
        if not os.path.exists(submodule_outputs_dir):
            os.mkdir(submodule_outputs_dir)

        if not os.path.exists(submodule_outputs_dir + '/train'):
            os.mkdir(submodule_outputs_dir + '/train')

        # Save the 3D numpy array to a pickle in submodule outputs dir
        for submodule_name, submodule_output_on_epoch in \
            zip(self.save_outputs_on_epoch.keys(), self.save_outputs_on_epoch.values()):

            submodule_output_3D = torch.stack((submodule_output_on_epoch))

            pickle_path = submodule_outputs_dir + '/train/' + submodule_name + '_output_on_epoch_end.pt'
            torch.save(submodule_output_3D, pickle_path)

        for submodule_name, submodule_dict in \
            zip(self.submodule_dicts.keys(), self.submodule_dicts.values()):

            if check_dict_key_exists('save_output_on_fit_end', submodule_dict):

                submodule_outputs = self(self.all_samples)

                submodule_output = submodule_outputs[submodule_name]

                submodule_output_arr = submodule_output.detach().numpy()

                filename = submodule_name + '_output_on_fit_end.csv'
                np.savetxt(submodule_outputs_dir + '/train/' + filename, 
                            submodule_output_arr, 
                            delimiter=',')

            if check_dict_key_exists('save_params', submodule_dict):

                # Make a directory to store the submodules
                submodules_dir = self.model_dir + '/submodule_params'
                if not os.path.exists(submodules_dir):
                    os.mkdir(submodules_dir)

                submodule_path = submodules_dir + '/' + submodule_name +'_params.pt'
                torch.save({'state_dict':self.submodules[submodule_name].state_dict()}, 
                            submodule_path)
