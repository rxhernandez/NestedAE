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

This repository contains code to inmplement NestedAE. It 
was created within the CONDA enviroment, and instructions 
for installing within it are available in user_md, though 
porting to other environnments (as long as the necessary
libraries are imported) should also be possible whout additional
code.

* For details of the datasets and how we trained NestedAE 
please refer to the paper; Thota, Sharma Priyadarshini,
Hernandez, Materials Horizons, submitted August 2023.

* For details on how to use NestedAE please read the 
user_guide.md in the docs folder.

* Any questions or comments please reach out via email
to the authors of the paper.

