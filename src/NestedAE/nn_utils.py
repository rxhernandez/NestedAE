" nn utilities script "

import math

from torch import nn, optim, zeros, matmul, Tensor, tensor, mean, sum
from torch import float32 as torch_float32
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, RichModelSummary # type: ignore
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme # type: ignore
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError # type: ignore

# Binary Linear layer implementation
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(Tensor(out_features, in_features).uniform_(-1, 1))
        if bias:
            self.bias = nn.Parameter(zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, X):

        if self.bias is not None:
            out = matmul(X, self.weight.clamp(min=0).sign().t()) + self.bias
        else:
            out = matmul(X, self.binary_weight.t())

        return out

def check_dict_key_exists(key, dictionary):
    """Check if key exists in dictionary

    Args:
        key (str) : key to check for
        dictionary (dict) : dictionary to check

    Returns: True if key exists in dictionary, False otherwise
    """
    if (key in list(dictionary.keys())):
        return True
    return False

def get_module_input_dim(connect_to, nn_params_dict, desc_shapes):
    """Get input dimension of a module.
    
    Args:
        connect_to (list) : list of modules/descriptors to connect to
        nn_params_dict (dict) : dictionary containing neural network parameters
        desc_shapes (dict) : dictionary containing shapes of descriptors
    
    Returns : tot_input_dim (int) : total input dimension of data to module
    """
    module_dicts = nn_params_dict['modules']
    tot_input_dim = 0
    for inp in connect_to:
        # Case where input to layer is the training data
        if inp in list(desc_shapes.keys()):
            input_dim = desc_shapes[inp][1]
        # Case where input to layer is ouput from last layer of connected module
        else:
            input_dim = module_dicts[inp]['output_dim']
        tot_input_dim += input_dim
    return tot_input_dim 

def set_layer_init(layer_list, module_dict, init='kernel'):
    """Initialize layer weights and biases.
    
    Args:
        layer_list (list) : list of torch.nn.Linear layers
        module_dict (dict) : dictionary containing module parameters
        init (str) : kernel/bias
    Returns :
        layer_list (list) : list of torch.nn.Linear layers with initialized weights and biases
    """
    layers = [layer for layer in layer_list \
                if isinstance(layer, nn.modules.linear.Linear)]
    # Calculating gain for xavier init of hidden layers
    if check_dict_key_exists('hidden_activation', module_dict):
        if module_dict['hidden_activation'] == 'relu':
            hidden_gain = math.sqrt(2)
        elif module_dict['hidden_activation'] == 'tanh':
            hidden_gain = 5.0 / 3
        else:
            hidden_gain = 1
    else:
        hidden_gain = 1
    # Calculating gain for xavier init of output layer
    if check_dict_key_exists('output_activation', module_dict):
        if module_dict['output_activation'] == 'relu':
            out_gain = math.sqrt(2)
        elif module_dict['output_activation'] == 'tanh':
            out_gain = 5.0 / 3
        else:
            out_gain = 1
    else:
        out_gain = 1

    if init == 'kernel':
        init_type = module_dict['layer_kernel_init']
    else:
        init_type = module_dict['layer_bias_init']

    for i, layer in enumerate(layers):
        if init == 'kernel':
            layer_params = layer.weight
        else:
            layer_params = layer.bias

        if '_' in init_type:
            scheme, distribution = init_type.split('_')[0], init_type.split('_')[1]
            # Use only with relu or leaky_relu
            if scheme == 'kaiming' and distribution == 'uniform':
                nn.init.kaiming_uniform_(layer_params, mode='fan_in')
            elif scheme == 'kaiming' and distribution == 'normal':
                nn.init.kaiming_normal_(layer_params, mode='fan_in')
            elif scheme == 'xavier' and distribution == 'uniform':
                if i == len(layers) - 1:
                    nn.init.xavier_uniform_(layer_params, gain=out_gain)
                    print(f' --> Setting out layer {init} init with {scheme} {distribution} distribution with gain {out_gain}')
                else:
                    nn.init.xavier_uniform_(layer_params, gain=hidden_gain)
                    print(f' --> Setting hidden layer {init} init with {scheme} {distribution} distribution with gain {out_gain}')
            else:
                if i == len(layers) - 1:
                    nn.init.xavier_normal_(layer_params, gain=out_gain)
                    print(f' --> Setting out layer {init} init with {scheme} {distribution} distribution with gain {out_gain}')
                else:
                    nn.init.xavier_normal_(layer_params, gain=hidden_gain)
                    print(f' --> Setting out layer {init} init with {scheme} {distribution} distribution with gain {out_gain}')
        elif init_type == 'normal':
            nn.init.normal_(layer_params, mean=0, std=1)
        elif init_type == 'uniform':
            nn.init.uniform_(layer_params, a=0, b=1)
        elif init_type == 'zeros':
            nn.init.zeros_(layer_params)
        else:
            raise ValueError(' --> Provided init scheme not among defined init schemes !')
        
    return layer_list

def create_scheduler_object(optimizer, nn_train_params_dict):
    """Create scheduler.
    
    Args:
        optimizer : optimizer object
        nn_train_params_dict : dictionary containing training parameters

    Returns : lr_scheduler_config (dict) : dictionary containing scheduler configuration
    """
    if nn_train_params_dict['scheduler']['type'] == 'expo':
        gamma = nn_train_params_dict['scheduler']['gamma']
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                                gamma=gamma,
                                                                verbose=True)
    elif nn_train_params_dict['scheduler']['type'] == 'step':
        step_size = nn_train_params_dict['scheduler']['step_size']
        gamma = nn_train_params_dict['scheduler']['gamma'] 
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                        step_size=step_size,
                                                        gamma=gamma,
                                                        verbose=True)
    elif nn_train_params_dict['scheduler']['type'] == 'multi_step':
        milestones = nn_train_params_dict['scheduler']['milestones']
        gamma = nn_train_params_dict['scheduler']['gamma']        
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=milestones,
                                                            gamma=gamma,
                                                            verbose=True)
    elif nn_train_params_dict['scheduler']['type'] == 'reduce_lr_on_plateau':
        mode = nn_train_params_dict['scheduler']['mode']
        factor = nn_train_params_dict['scheduler']['factor']
        patience = nn_train_params_dict['scheduler']['patience']
        cooldown = nn_train_params_dict['scheduler']['cooldown']
        min_lr = nn_train_params_dict['scheduler']['min_lr']
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                    mode=mode,
                                                                    factor=factor,
                                                                    patience=patience,
                                                                    cooldown=cooldown,
                                                                    min_lr=min_lr,
                                                                    verbose=True)   
    else:
        raise ValueError(' --> Provided Learning Rate Scheduling scheme has not been defined.')
    lr_scheduler_config = {# REQUIRED: The scheduler instance
                            "scheduler": lr_scheduler,
                            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                            "monitor": nn_train_params_dict['scheduler']['monitor'],
                            "frequency": nn_train_params_dict['scheduler']['frequency'],
                            # If set to `True`, will enforce that the value specified 'monitor'
                            # is available when the scheduler is updated, thus stopping
                            # training if not found. If set to `False`, it will only produce a warning
                            "strict": True,
                            # If using the `LearningRateMonitor` callback to monitor the
                            # learning rate progress, this keyword can be used to specify
                            # a custom logged name
                            "name": nn_train_params_dict['scheduler']['name']}
    return lr_scheduler_config

# Class that creates the TF optimizer object            
def create_optimizer_object(modules, nn_train_params_dict):
    """Create optimizer.
    
    Args:
        modules (torch.nn.ModuleDict()) : dictionary containing torch modules
        nn_train_params_dict : dictionary containing training parameters

    Returns : optimizer_object (torch.optim)
    """
    optimizer_type = nn_train_params_dict['optimizer']['type']
    # Check if per module lr is required 
    if check_dict_key_exists('module_name', nn_train_params_dict['optimizer']):
        params_groups = []
        for i, module_name in enumerate(nn_train_params_dict['optimizer']['module_name']):
            module_params = {}
            module_params['params'] = modules[module_name].parameters()
            module_params['lr'] = nn_train_params_dict['optimizer']['lr'][i]
            params_groups.append(module_params)
        if optimizer_type == 'adam':
            adam_optimizer = optim.Adam(params_groups,
                                        betas=(0.9, 0.999),
                                        eps=1e-8,
                                        weight_decay=0,
                                        amsgrad=False)
            return adam_optimizer
        elif optimizer_type == 'sgd':
            sgd_optimizer = optim.SGD(params_groups,
                                      momentum=nn_train_params_dict['optimizer']['momentum'])
            return sgd_optimizer
        else:
            raise ValueError(f' --> {optimizer_type} has not been defined.')
    else: # One optimizer for all parameters
        module_params = modules.parameters()
        if optimizer_type == 'adam':
            adam_optimizer = optim.Adam(module_params,
                                        lr=nn_train_params_dict['optimizer']['lr'],
                                        betas=(0.9, 0.999),
                                        eps=1e-8,
                                        weight_decay=0,
                                        amsgrad=False)
            return adam_optimizer
        elif optimizer_type == 'sgd':
            sgd_optimizer = optim.SGD(module_params,
                                      lr=nn_train_params_dict['optimizer']['lr'],
                                      momentum=nn_train_params_dict['optimizer']['momentum'])
            return sgd_optimizer
        else:
            raise ValueError(f' --> {optimizer_type} has not been defined.')

def create_callback_object(nn_train_params_dict, nn_save_dir):
    """Create callback.
    
    Args:
        nn_train_params_dict (dict) : dictionary containing training parameters
        nn_save_dir (str) : directory to save model checkpoints

    Returns:
        callback_objects (list) : list of callback objects
    """
    callback_objects = []
    callback_types = list(nn_train_params_dict['callbacks'].keys())
    callback_dicts = list(nn_train_params_dict['callbacks'].values())
    for i, callback_dict in enumerate(callback_dicts):
        callback_type = callback_types[i]
        if callback_type == 'early_stopping':
            early_stopping = EarlyStopping(monitor=callback_dict['monitor'],
                                            min_delta=callback_dict['min_delta'],
                                            patience=callback_dict['patience'],
                                            verbose=True,
                                            mode=callback_dict['mode'],
                                            strict=True,
                                            check_finite=True)
            callback_objects.append(early_stopping)
        elif callback_type == 'model_checkpoint':
            checkpoints_dir = nn_save_dir + '/checkpoints'
            model_checkpoint = ModelCheckpoint(dirpath=checkpoints_dir,
                                                filename='{epoch}-{total_val_loss:.2f}',
                                                monitor=callback_dict['monitor'],
                                                verbose=False,
                                                save_last=True,
                                                save_top_k=callback_dict['save_top_k'],
                                                mode=callback_dict['mode'],
                                                auto_insert_metric_name=True)
            callback_objects.append(model_checkpoint)
        elif callback_type == 'rich_model_summary':
            callback_objects.append(RichModelSummary(max_depth=1))
        elif callback_type == 'rich_progress_bar':
            progress_bar = RichProgressBar(
                            theme=RichProgressBarTheme(
                                description="green_yellow",
                                progress_bar="green1",
                                progress_bar_finished="green1",
                                progress_bar_pulse="#6206E0",
                                batch_progress="green_yellow",
                                time="grey82",
                                processing_speed="grey82",
                                metrics="grey82"),
                            leave=False)
            callback_objects.append(progress_bar)
        else:
            raise ValueError(f' --> {callback_type} callback not defined.')
    return callback_objects