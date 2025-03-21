Nested Autotoencoders (NestedAEs)
----------------

Multiscale modelling involves inferring physics 
at a given spatial and temporal scale based on the
physics at a finer/smaller scale. This is done under the assumption
that the finer scale physics are better understood than the
coarser scale physics. In this work, we developed
a novel neural network model called 
Nested Autoencoders (NestedAE) to extract
**important** physical features and predict properties
at a given length scale and correlate them 
with properties predicted on a larger length scale.

While this idea is general and can be applied to
any system that displays distinct characteristics
and properties at different length scales, we 
demonstrated the application of this model on 

(1) a synthetic dataset created from nested analytical
functions whose dimensionality is therefore known a priori,
and 

(2) a multi-scale metal halide perovskite dataset that is the 
combination of two open source datasets containing 
atomic and ionic properties, and device characterization 
using JV analysis, respectively.


<hr>

Installation
----------------

Clone the repository to the directory of your choice on your local machine.

```bash
git clone git@github.com:T-NIKHIL/NestedAE.git
```

One-step installation :

** FYI ** : The script is designed to be run on MacOS systems. For other
operating systems please refer to the manual installation section below 
and find the appropriate terminal commands for your OS to execute each step.

```bash
./install.sh
```

To manually install, please follow the step-by-step instructions below :

Manual installation

* Create a new virtual environment to install the NestedAE library using
either Conda or python venv.

* Using Conda :

    * This method requires you to have already installed conda on your local machine.
    For more details please visit the [Conda website](https://docs.anaconda.com/miniconda/install/).

    * Create a new conda environment with python=3.9 

    ```bash 
    conda create -n NestedAE python=3.9
    ```

* Using python venv :

    * This is a barebones method of creating a python virtual environment and
    is what is followed in install.sh script.  

    ```bash
    python3.9 -m venv python_venvs/NestedAE
    ```

* After running the above command, you will see the NestedAE directory
contains a new directory called "python_venvs" and inside it is the
"NestedAE" virtual environment directory. To activate this virtual 
environment go back to the project root directory and type the following
command. To deactivate the virtual environment, type `deactivate`.

```bash
source python_venvs/NestedAE/bin/activate
```

* The name of the virtual environment will be displayed in the terminal
and typing `pip list` will show that *pip* and *setuptools* python libraries
are installed by default. 

* We will need a *build frontend* tool to install
the NestedAE python package in the NestedAE virtual environment. 
For this we will download the *build* python package.

```bash
pip install -U build
```

* Create the NestedAE *wheel* file.

```bash
python -m build --wheel
```

**Optional** : You can also create a source distribution (which is essentially a zipped file containing the source code) by typing : `python -m build --sdist`. 

* Now we can finally install the NestedAE python package. 

```bash
pip install .
```

* Voilá ! You have now successfully installed the NestedAE python package. You can
now run pytest to check for successful installation.

```bash
pytest
```

Documentation
----------------

* For details of the datasets and how we trained NestedAE 
please refer to the paper, noted in the "Citing" section below.

* For details on how to use NestedAE please refer to the 
[user_guide.md](https://github.com/rxhernandez/NestedAE/blob/main/user_guide.md) in the docs folder.

* Any questions or comments please reach out via email
to the authors of the paper.


<hr>

Authors
----------------

The NestedAE codes and databaess were developed by Nikhil K. Thota, Maitreyee Sharma Priyadarshini and Rigoberto Hernandez

Contributors can be found [here](https://github.com/rxhernandez/NestedAE/graphs/contributors).

<hr>

Citing
----------------

If you use database or codes, please cite the paper:

>N. K. Thota, M. Sharma Priyadarshini and R. Hernandez, “NestedAE: Interpretable Nested Autoencoders for Multi-Scale Material Modelling,” _Mater. Horiz._, **11**, 700, (2024). [(0.1039/D3MH01484C)](http://doi.org/10.1039/D3MH01484C)

and/or this site:

>N. K. Thota, M. Sharma Priyadarshini and R. Hernandez, NestedAE, URL, [https://github.com/rxhernandez/NestedAE](https://github.com/rxhernandez/NestedAE)

<hr>

Acknowledgment
----------------

This work was supported by 
the Department of Energy (DOE), Office of Science, Basic Energy Science (BES), under Award #DE-SC0022305.


<hr>

License
----------------

NestedAE code and databases are distributed under terms of the [MIT License](https://github.com/rxhernandez/NestedAE/blob/main/LICENSE).

