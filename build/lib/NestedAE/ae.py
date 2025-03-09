"AE class"

import sys
import os
import pprint

from numpy import savetxt
from torch import save, stack, load, nn, concatenate, tensor
from torch import float32 as torch_float32
from pytorch_lightning import LightningModule # type: ignore

# User defined libraries
from NestedAE import nn_utils
from NestedAE.nn_utils import check_dict_key_exists

class AE(LightningModule):
    """Class for building autoencoders using linear layers."""

    def __init__(self,
                 ae_save_dir_path,
                 nn_params_dict,
                 nn_train_params_dict,
                 dataset_path):

        super(AE, self).__init__()

        # Save the parameters passed into __init__
        self.save_hyperparameters()
        self.automatic_optimization = True
        self.name = nn_params_dict['name']
        self.ae_save_dir_path = ae_save_dir_path
        self.nn_params_dict = nn_params_dict
        self.nn_train_params_dict = nn_train_params_dict
        self.datasets = load(dataset_path, weights_only=False)
        self.all_samples = self.datasets[:]
        self.example_input = self.datasets[0]
        self.ae_module_dicts = nn_params_dict['modules']
        self.ae_module_losses = None
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

        ae_modules = {}
        # Outer loop iterates over the ae_modules
        for ae_module_name, ae_module_dict in self.ae_module_dicts.items():
            layer_list = nn.ModuleList()
            # Check for existence of keys or take defualts if not present
            if check_dict_key_exists('hidden_layers', ae_module_dict):
                hidden_layers = ae_module_dict['hidden_layers']
            else:
                hidden_layers = 0
            if check_dict_key_exists('hidden_dim', ae_module_dict):
                hidden_dim = ae_module_dict['hidden_dim']
            else:
                hidden_dim = None
            if check_dict_key_exists('hidden_activation', ae_module_dict):
                hidden_activation = ae_module_dict['hidden_activation']
            else:
                hidden_activation = None
            if check_dict_key_exists('output_activation', ae_module_dict):
                output_activation = ae_module_dict['output_activation']
            else:
                output_activation = None
            if check_dict_key_exists('layer_dropout', ae_module_dict):
                layer_dropout = ae_module_dict['layer_dropout']
            else:
                layer_dropout = None
            if check_dict_key_exists('layer_kernel_init', ae_module_dict):
                layer_kernel_init = ae_module_dict['layer_kernel_init']
            else:
                layer_kernel_init = None
            if check_dict_key_exists('layer_bias_init', ae_module_dict):
                layer_bias_init = ae_module_dict['layer_bias_init']
            else:
                layer_bias_init = None
            if check_dict_key_exists('load_params', ae_module_dict):
                load_params = ae_module_dict['load_params']
            else:
                load_params = False

            num_layers = hidden_layers + 1
            for layer_num in range(num_layers):
                if layer_num == 0:
                    # Calculate the input dimensions to first layer
                    input_dim = nn_utils.get_module_input_dim(ae_module_dict['connect_to'],
                                                                self.nn_params_dict,
                                                                self.datasets.desc_shapes)
                    if hidden_dim is not None:
                        layer_list.append(nn.Linear(in_features=input_dim,
                                                out_features=ae_module_dict['hidden_dim'],
                                                bias=True))
                    else:
                        layer_list.append(nn.Linear(in_features=input_dim,
                                                out_features=ae_module_dict['output_dim'],
                                                bias=True))
                        if output_activation:
                            layer_list.append(output_activation)
                        break # Only output layer
                elif layer_num == num_layers - 1:
                    layer_list.append(nn.Linear(in_features=ae_module_dict['hidden_dim'],
                                                out_features=ae_module_dict['output_dim'],
                                                bias=True))
                    if output_activation:
                        layer_list.append(output_activation)
                    break # Dont add hidden activations
                else:
                    layer_list.append(nn.Linear(in_features=ae_module_dict['hidden_dim'],
                                                out_features=ae_module_dict['hidden_dim'],
                                                bias=True))
                # Add hidden activations if specified
                if hidden_activation:
                    layer_list.append(hidden_activation)
                if layer_dropout:
                    layer_list.append(layer_dropout)
            # Initialize weights for all layers
            if layer_kernel_init:
                layer_list = nn_utils.set_layer_init(layer_list, ae_module_dict, init='kernel')
            if layer_bias_init:
                layer_list = nn_utils.set_layer_init(layer_list, ae_module_dict, init='bias')
            # Check to see if a ae_module has to be loaded
            if load_params:
                try:
                    ae_module_params = load(load_params, weights_only=False)
                    layer_list.load_state_dict(ae_module_params['state_dict'])
                    print(f' --> Loaded ae_module {ae_module_name}.')
                except OSError as err:
                    raise FileNotFoundError(f' --> Could not load {ae_module_name}.') from err
            # Finally add to ae_module list
            ae_modules[ae_module_name] = layer_list
            print('\n')
            print(f' --> ae_module {ae_module_name} layers :')
            print(layer_list)
        self.ae_modules = nn.ModuleDict(ae_modules)

    def forward(self, ae_inputs):
        """Forward pass through the model."""
        # Stores all module outputs
        ae_module_outputs = {}
        ae_input_ids = list(ae_inputs.keys())

        # Outer loop iterates over the modules
        for ae_module_name, ae_module in self.ae_modules.items():
            # Get the input ids that are connected to the ae_module
            ae_module_input_ids = self.ae_module_dicts[ae_module_name]['connect_to']
            inp = []
            for ae_module_input_id in ae_module_input_ids:
                if ae_module_input_id in ae_input_ids:
                    inp.append(ae_inputs[ae_module_input_id])
                else:  # Then input is output from previous module
                    inp.append(ae_module_outputs[ae_module_input_id])
            # Convert list of tensors to a single tensor
            inp = concatenate(inp, dim=-1)
            for j, layer in enumerate(ae_module):
                if j == 0:
                    output = layer(inp)
                else:
                    output = layer(output)
            ae_module_outputs[ae_module_name] = output
            if self.trace_model:
                print('\n')
                print(' ---------------------------------- ')
                print(f'ae_module_name:{ae_module_name}')
                print(f'input id:{ae_module_input_ids}')
                print('input to module :')
                print(inp)
                print('output from module :')
                print(output)
                print('module output dictionary :')
                pp = pprint.PrettyPrinter()
                pp.pprint(ae_module_outputs)
                print(' ---------------------------------- ')
                print('\n')
        return ae_module_outputs

    # <-- TESTING -->
    # Uncomment for using custom pytorch training loops
    # def backward(self, loss, optimizer):
    #    loss.backward()

    def compile(self):
        """Compile the model."""
        losses = {}
        for ae_module_name, ae_module_dict in self.ae_module_dicts.items():
            if 'loss' in list(ae_module_dict.keys()):
                losses[ae_module_name] = ae_module_dict['loss']['type']
        self.ae_module_losses = losses

    def configure_optimizers(self):
        """Configure the optimizers."""
        optimizer = nn_utils.create_optimizer_object(self.ae_modules, self.nn_train_params_dict)
        if 'scheduler' in list(self.nn_train_params_dict.keys()):
            scheduler = nn_utils.create_scheduler_object(optimizer, self.nn_train_params_dict)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {'optimizer': optimizer}

    def training_step(self, batch, batch_idx):
        """Training step."""
        # Pass data into model
        ae_module_outputs = self(batch)
        train_loss_values = {}
        # Init total loss to 0
        total_train_loss = tensor(0, device=self.device, dtype=torch_float32)
        # Prediction losses
        for ae_module_name, ae_module_loss in self.ae_module_losses.items():
            # Output from module
            output = ae_module_outputs[ae_module_name]
            target_name = self.ae_module_dicts[ae_module_name]['loss']['target']
            target = batch[target_name]

            loss_name = self.ae_module_dicts[ae_module_name]['loss']['type'].__class__.__name__
            loss_wt = self.ae_module_dicts[ae_module_name]['loss']['wt']
            loss_wt = tensor(loss_wt, device=self.device, dtype=torch_float32, requires_grad=False)
            if 'sample_wts' in list(batch.keys()):
                sample_wt = batch['sample_wts']
            else:
                sample_wt = tensor(1, device=self.device, dtype=torch_float32, requires_grad=False)

            pred_loss = ae_module_loss(output, target).multiply(loss_wt).multiply(sample_wt)
            total_train_loss = total_train_loss.add(pred_loss)
            train_loss_values['train_' + target_name + '_' + loss_name] = pred_loss.item()

            if 'metric' in list(self.ae_module_dicts[ae_module_name].keys()):
                for metric in self.ae_module_dicts[ae_module_name]['metric']:
                    metric_train = metric(output, target)
                    metric_name = metric.__class__.__name__
                    train_loss_values['train_' + target_name + '_' + metric_name] = metric_train.item()

        # Init Regularization losses
        l1_param_loss = tensor(0, device=self.device, dtype=torch_float32)
        l2_param_loss = tensor(0, device=self.device, dtype=torch_float32)

        # Regularization losses
        for ae_module_name, ae_module_dict in self.ae_module_dicts.items():

            if check_dict_key_exists('layer_weight_reg_l1', ae_module_dict):
                lambda_l1 = ae_module_dict['layer_weight_reg_l1']
            else:
                lambda_l1 = 0

            if check_dict_key_exists('layer_weight_reg_l2', ae_module_dict):
                lambda_l2 = ae_module_dict['layer_weight_reg_l2']
            else:
                lambda_l2 = 0

            if lambda_l1 > 0:
                lambda_l1 = tensor(lambda_l1, device=self.device, dtype=torch_float32, requires_grad=False)
                for name, params in self.ae_modules[ae_module_name].named_parameters():
                    if name.endswith('.weight') or name.endswith('.bias'):
                        p = params.view(-1)
                        l1_param_loss = l1_param_loss.add(p.abs().sum().multiply(lambda_l1))

            if lambda_l2 > 0:
                lambda_l2 = tensor(lambda_l2, device=self.device, dtype=torch_float32, requires_grad=False)
                for name, params in self.ae_modules[ae_module_name].named_parameters():
                    if name.endswith('.weight') or name.endswith('.bias'):
                        # Params view is important here since weights is a 2D tensor which we unwrap to a 1D tensor
                        # params.data :- returns the weight data. No reshape
                        # params.view :- returns the weight data. With reshape
                        p = params.view(-1)
                        l2_param_loss = l2_param_loss.add(p.pow(2).sum().multiply(lambda_l2))

        # Add in regularization losses
        total_train_loss += l1_param_loss + l2_param_loss
        # The weighted loss
        train_loss_values['total_train_loss'] = total_train_loss.item()
        train_loss_values['l1_param_loss'] = l1_param_loss.item()
        train_loss_values['l2_param_loss'] = l2_param_loss.item()
        self.train_loss_values = train_loss_values
        self.log_dict(train_loss_values, on_step=False,
                      on_epoch=True, logger=True, prog_bar=False,
                      batch_size=self.nn_train_params_dict['batch_size'])
        return total_train_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Pass data into model
        ae_module_outputs = self(batch)
        # Init total loss to 0
        total_val_loss = tensor(0, device=self.device, dtype=torch_float32)
        val_loss_values = {}
        # Prediction losses
        for ae_module_name, ae_module_loss in self.ae_module_losses.items():
            # Output from module
            output = ae_module_outputs[ae_module_name]
            target_name = self.ae_module_dicts[ae_module_name]['loss']['target']
            target = batch[target_name]
            
            loss_name = self.ae_module_dicts[ae_module_name]['loss']['type'].__class__.__name__
            loss_wt = self.ae_module_dicts[ae_module_name]['loss']['wt']
            loss_wt = tensor(loss_wt, device=self.device, dtype=torch_float32, requires_grad=False)
            if 'sample_wts' in list(batch.keys()):
                sample_wt = batch['sample_wts']['wts']
                sample_wt = tensor(sample_wt, device=self.device, dtype=torch_float32)
            else:
                sample_wt = tensor(1, device=self.device, dtype=torch_float32)
            
            pred_loss = ae_module_loss(output, target)*loss_wt*sample_wt
            total_val_loss += pred_loss
            val_loss_values['val_' + target_name + '_' + loss_name] = pred_loss.item()

            if 'metric' in list(self.ae_module_dicts[ae_module_name].keys()):
                for metric in self.ae_module_dicts[ae_module_name]['metric']:
                    metric_val = metric(output, target)
                    metric_name = metric.__class__.__name__
                    val_loss_values['val_' + target_name + '_' + metric_name] = metric_val.item()

        val_loss_values['total_val_loss'] = total_val_loss.item()
        self.val_loss_values = val_loss_values
        self.log_dict(val_loss_values,
                      on_step=False,
                      on_epoch=True,
                      logger=True,
                      prog_bar=False,
                      batch_size=self.nn_train_params_dict['batch_size'])

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

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Store the module outputs
        ae_module_outputs = self(self.all_samples)
        for ae_module_name, ae_module_dict in self.ae_module_dicts.items():
            if 'save_output_on_epoch_end' in list(ae_module_dict.keys()):
                if ae_module_dict['save_output_on_epoch_end']:
                    ae_module_output = ae_module_outputs[ae_module_name]
                    if self.current_epoch == 0:
                        self.save_outputs_on_epoch[ae_module_name] = [ae_module_output]
                    else:
                        self.save_outputs_on_epoch[ae_module_name].append(ae_module_output)

    def on_fit_end(self):
        """Called at the end of fit() to do things such as logging, saving etc."""
        for ae_module_name, ae_module_dict in self.ae_module_dicts.items():
            if 'save_output_on_fit_end' in list(ae_module_dict.keys()):
                if ae_module_dict['save_output_on_fit_end']:
                    # Create a module outputs dir if it does not exist
                    ae_module_outputs_dir = f'{self.ae_save_dir_path}/ae_module_outputs'
                    if not os.path.exists(ae_module_outputs_dir):
                        os.mkdir(ae_module_outputs_dir)
                    if not os.path.exists(ae_module_outputs_dir + '/train'):
                        os.mkdir(ae_module_outputs_dir + '/train')
                    ae_module_outputs = self(self.all_samples)
                    ae_module_output = ae_module_outputs[ae_module_name]
                    ae_module_output_arr = ae_module_output.detach().numpy()
                    filename = ae_module_name + '_output_on_fit_end.csv'
                    savetxt(ae_module_outputs_dir + '/train/' + filename,
                            ae_module_output_arr,
                            delimiter=',')
            if 'save_output_on_epoch_end' in list(ae_module_dict.keys()):
                if ae_module_dict['save_output_on_epoch_end']:
                    # Create a module outputs dir if it does not exist
                    ae_module_outputs_dir = f'{self.ae_save_dir_path}/ae_module_outputs'
                    if not os.path.exists(ae_module_outputs_dir):
                        os.mkdir(ae_module_outputs_dir)
                    if not os.path.exists(ae_module_outputs_dir + '/train'):
                        os.mkdir(ae_module_outputs_dir + '/train')
                    # Save the 3D numpy array to a pickle in module outputs dir
                    for ae_module_name, ae_module_output_on_epoch in self.save_outputs_on_epoch.items():
                        ae_module_output_3D = stack((ae_module_output_on_epoch))
                        pickle_path = ae_module_outputs_dir + '/train/' + ae_module_name + '_output_on_epoch_end.pt'
                        save(ae_module_output_3D, pickle_path)
            if 'save_params' in list(ae_module_dict.keys()):
                if ae_module_dict['save_params']:
                    # Make a directory to store the modules
                    ae_modules_dir = f'{self.ae_save_dir_path}/ae_module_params'
                    if not os.path.exists(ae_modules_dir):
                        os.mkdir(ae_modules_dir)
                    ae_module_path = f'{ae_modules_dir}/{ae_module_name}_params.pt'
                    save({'state_dict': self.ae_modules[ae_module_name].state_dict()}, ae_module_path)