#!/bin/bash

## source ~/miniconda/bin/activate

# <-- USER START -->

## Specify folder to look at for the input files
inputs_dir=inputs_perov_data
## Specify folder to save the NestedAE outputs to
run_dir=test_nt
## Which Autoencoder(ae) to use for training or inference
ae=1
## Specify mode of operation. train or predict
mode=train
## Specify accelerator to use. cpu or gpu
accelerator=cpu
## Name of submodule from which predictions are required.
## Use when selecting 'predict' mode. Ignored in 'train' mode.
submodule=bg_pred

# <-- USER END -->

if [[ ! -d ../runs ]]; then
    mkdir ../runs
fi

## The inputs directory can be any name.
## The code copies inputs_dir to inputs.
rm -rf inputs
cp -rf $inputs_dir inputs

## Uncomment '##' for to skip preprocessing. By default, this section of code will run preprocessing of data for NestedAE
##<<comment
python3 preprocess_data.py --run_dir $run_dir --ae $ae --mode $mode &
wait
rm -rf inputs
##comment

## Uncomment '##' to skip NestedAE training. By default, this section of code will run NestedAE
##<<comment
if [[ $mode == train ]]; then
	echo "In train mode"
	python3 train.py --run_dir $run_dir --ae $ae --accelerator $accelerator
fi 
if [[ $mode == predict ]]; then
	echo "In predict mode"
	python3 predict.py --run_dir $run_dir --ae $ae --accelerator $accelerator --submodule $submodule
fi
##comment


