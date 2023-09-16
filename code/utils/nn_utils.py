" nn utilities script "

import torch
from torch import nn

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, RichModelSummary
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

################################################################################################
# Add Custom Layer Implementations here
################################################################################################

# Binary Linear layer implementation
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-1, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, X):

        if self.bias is not None:
            out = torch.matmul(X, self.weight.clamp(min=0).sign().t()) + self.bias
        else:
            out = torch.matmul(X, self.binary_weight.t())

        return out

################################################################################################

def check_dict_key_exists(key, dictionary):
    """Check if key exists in dictionary."""
    if (key in list(dictionary.keys())) and \
        (dictionary[key] is not None):
        return True
    return False

def create_loss_object(loss):
    """Create loss object."""
    if loss == 'mae':
        loss_obj = nn.L1Loss()
    elif loss == 'mse':
        loss_obj = nn.MSELoss()
    elif loss == 'huber':
        loss_obj = nn.HuberLoss()
    elif loss == 'hinge':
        loss_obj = nn.HingeEmbeddingLoss()
    elif loss == 'kld':
        loss_obj = nn.KLDivLoss()
    elif loss == 'nll':
        loss_obj = nn.NLLLoss(reduction='mean')
    elif loss == 'ce':
        loss_obj = nn.CrossEntropyLoss(reduction='mean')
    elif loss == 'bce':
        # Reduction : Mean (The mean loss of a batch is calculated)
        # Reduction : Sum (The loss is summed over the batch)
        loss_obj = nn.BCELoss(reduction='mean')
    elif loss == 'bcewithlogits':
        # Reduction : Mean (The mean loss of a batch is calculated)
        # Reduction : Sum (The loss is summed over the batch)
        loss_obj = nn.BCEWithLogitsLoss(reduction='mean')
    elif loss == 'name_of_loss':
        #tf_loss = nameOfLoss()
        raise ValueError(' --> Loss Not Implemented !')
    else:
        raise ValueError(' --> Not among TF or user defined losses !')

    return loss_obj

def get_module_input_dim(connect_to, nn_params_dict, variable_shapes):
    """Get input dimension of a module."""
    submodule_dicts = nn_params_dict['submodules']

    tot_input_dim = 0

    for inp in connect_to:

        # Case where input to layer is the training data
        if inp in list(variable_shapes.keys()):
            input_dim = variable_shapes[inp][1]
        # Case where input to layer is ouput from last layer of connected submodule
        else:
            if isinstance(submodule_dicts[inp]['num_nodes_per_layer'], list):
                input_dim = submodule_dicts[inp]['num_nodes_per_layer'][-1]
            else:
                input_dim = submodule_dicts[inp]['num_nodes_per_layer']

        tot_input_dim += input_dim

    return tot_input_dim 

def set_layer_init(submodule, activation, kernel_init_type, kernel_init_gain, bias_init_type):
    """Initialize layer weights and biases."""
    linear_layers = [layer for layer in submodule \
                    if isinstance(layer, torch.nn.modules.linear.Linear)]

    for i, linear_layer in enumerate(linear_layers):

        init_type = kernel_init_type[i]
        gain = kernel_init_gain[i]
        if '_' in init_type:
            scheme, distribution = init_type.split('_')[0], init_type.split('_')[1]
            # Use only with relu or leaky_relu
            if scheme == 'kaiming' and distribution == 'uniform':
                torch.nn.init.kaiming_uniform_(linear_layer.weight,\
                                                mode='fan_in',\
                                                nonlinearity=activation[i])
            elif scheme == 'kaiming' and distribution == 'normal':
                if activation[i] == 'selu':
                    nonlinearity = 'linear'
                else:
                    nonlinearity = activation[i]
                torch.nn.init.kaiming_normal_(linear_layer.weight,\
                                                mode='fan_in',\
                                                nonlinearity=nonlinearity)
            elif scheme == 'xavier' and distribution == 'uniform':
                torch.nn.init.xavier_uniform_(linear_layer.weight,\
                                                gain=gain)
            elif scheme == 'xavier' and distribution == 'normal':
                torch.nn.init.xavier_normal_(linear_layer.weight,\
                                                gain=gain)
        else:
            if kernel_init_type[i] == 'normal':
                torch.nn.init.normal_(linear_layer.weight, mean=0, std=1)
            elif kernel_init_type[i] == 'uniform':
                torch.nn.init.uniform_(linear_layer.weight, a=0, b=1)
            else:
                raise ValueError(' --> Provided weight init scheme not among defined kernel init schemes !')

    # Initialize bias from one of the simple initialization schemes

    for i, linear_layer in enumerate(linear_layers):
        if bias_init_type[i] == 'zeros':
            torch.nn.init.zeros_(linear_layer.bias)
        else:
            raise ValueError(' --> Provided bias init scheme not among defined bias init schemes !')

    return submodule

def set_layer_activation(activation):
    """Set layer activation function."""

    # The ReLU family
    if activation == 'relu':
        act_obj = nn.ReLU()
    elif activation == 'leaky_relu':
        act_obj = nn.LeakyReLU(negative_slope=0.5)
    elif activation == 'elu':
        act_obj = nn.ELU(alpha=1.0)
    elif activation == 'prelu':
        act_obj = nn.PReLU(num_parameters=1,
                            init=0.25)
    elif activation == 'selu':
        act_obj = nn.SELU()
    elif activation == 'sigmoid':
        act_obj = nn.Sigmoid()
    elif activation == 'softmax':
        act_obj = nn.Softmax(dim=-1)
    elif activation == 'logsoftmax':
        act_obj = nn.LogSoftmax(dim=-1)
    elif activation == 'softsign':
        act_obj = nn.Softsign()
    elif activation == 'tanh':
        act_obj = nn.Tanh()
    elif activation == 'hardtanh':
        act_obj = nn.Hardtanh(min_val=-2, max_val=2)
    elif activation == 'tanhshrink':
        act_obj = nn.Tanhshrink()
    elif activation == 'softplus':
        act_obj = nn.Softplus(beta=5, threshold=20)
    elif activation == 'silu':
        act_obj = nn.SiLU()
    else:
        raise ValueError(f' --> {activation} not among defined activation functions !')

    return act_obj

def set_layer_dropout(dropout_type, p):
    """Set layer dropout."""
    if dropout_type == 'Dropout':
        drop_obj = nn.Dropout(p=p)
    elif dropout_type == 'AlphaDropout':
        drop_obj = nn.AlphaDropout(p=p)
    else:
        raise ValueError(f' --> {dropout_type} not among defined dropout functions !')
    
    return drop_obj

def create_scheduler_object(optimizer, nn_train_params_dict):
    """Create scheduler."""
    if nn_train_params_dict['scheduler']['type'] == 'expo':
        gamma = nn_train_params_dict['scheduler']['gamma']
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                                gamma=gamma,
                                                                verbose=True)

    if nn_train_params_dict['scheduler']['type'] == 'step':
        step_size = nn_train_params_dict['scheduler']['step_size']
        gamma = nn_train_params_dict['scheduler']['gamma'] 
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                        step_size=step_size,
                                                        gamma=gamma,
                                                        verbose=True)

    if nn_train_params_dict['scheduler']['type'] == 'multi_step':
        milestones = nn_train_params_dict['scheduler']['milestones']
        gamma = nn_train_params_dict['scheduler']['gamma']        
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=milestones,
                                                            gamma=gamma,
                                                            verbose=True)
        
    if nn_train_params_dict['scheduler']['type'] == 'reduce_lr_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=40,
                                                                    cooldown=10,
                                                                    min_lr=1e-8,
                                                                    verbose=True)   

    else:
        raise ValueError(' --> Provided Learning Rate Scheduling scheme has not been defined.')


    lr_scheduler_config = {# REQUIRED: The scheduler instance
                            "scheduler": lr_scheduler,
                            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                            "monitor": "total_val_loss",
                            "frequency": 1,
                            # If set to `True`, will enforce that the value specified 'monitor'
                            # is available when the scheduler is updated, thus stopping
                            # training if not found. If set to `False`, it will only produce a warning
                            "strict": True,
                            # If using the `LearningRateMonitor` callback to monitor the
                            # learning rate progress, this keyword can be used to specify
                            # a custom logged name
                            "name": None}

    return lr_scheduler_config

# Class that creates the TF optimizer object            
def create_optimizer_object(submodules, nn_train_params_dict):
    """Create optimizer."""

    optimizer_type = nn_train_params_dict['optimizer']['type']

    if optimizer_type == 'adam':

        # Check if per parameter optimizer is required 
        if check_dict_key_exists('submodule_name', nn_train_params_dict['optimizer']) is True:

            module_params = []
            for i, submodule_name in enumerate(nn_train_params_dict['optimizer']['submodule_name']):
                submodule_params = {}
                submodule_params['params'] = submodules[submodule_name].parameters()
                submodule_params['lr'] = nn_train_params_dict['optimizer']['lr'][i]
                module_params.append(submodule_params)

            adam_optimizer = torch.optim.Adam(module_params,
                                            betas=(0.9, 0.999),
                                            eps=1e-8,
                                            weight_decay=0,
                                            amsgrad=False)

        # One optimizer for all parameters
        else:
            module_params = submodules.parameters()

            adam_optimizer = torch.optim.Adam(module_params,
                                            lr=nn_train_params_dict['optimizer']['lr'],
                                            betas=(0.9, 0.999),
                                            eps=1e-8,
                                            weight_decay=0,
                                            amsgrad=False)

        return adam_optimizer
    
    raise ValueError(f' --> {optimizer_type} has not been defined.')

def create_callback_object(nn_train_params_dict, nn_save_dir):
    """Create callback."""

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
                                                verbose=True,
                                                save_last=True,
                                                save_top_k=callback_dict['save_top_k'],
                                                mode=callback_dict['mode'],
                                                auto_insert_metric_name=True)
            callback_objects.append(model_checkpoint)

        else:
            raise ValueError(f' --> {callback_type} callback not defined.')

    callback_objects.append(RichModelSummary(max_depth=1))

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

    return callback_objects
