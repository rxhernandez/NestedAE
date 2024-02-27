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

Documentation
----------------

This repository contains code to implement NestedAE. It 
was created within the CONDA enviroment, and instructions 
for installing within it are available in the [user guide](https://github.com/rxhernandez/NestedAE/blob/main/user_guide.md), though 
porting to other environnments (as long as the necessary
libraries are imported) should also be possible whout additional
code.

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

