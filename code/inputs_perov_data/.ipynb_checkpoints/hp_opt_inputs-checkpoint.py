# Parameters for keras tuner - Bayesian optimization 
keras_tuner_bayes_opt_dict={
    'objective':'val_loss',
    'max_trials':10,
    'num_initial_points':2,
    'alpha':0.0001,
    'beta':2.6,
}