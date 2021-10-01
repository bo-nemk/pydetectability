# Implementation of Taal distortion detectability
Implementation is based on the original paper "A Low-complexity spectro-temporal based perceptual model" and the code provided on https://ceestaal.nl/code/.

## Model Implementation
The model by Taal et al. proposes a "perceptual distance measure" to quantify the perceived difference between a clean and degraded audio stimuli. This perceptual distance is given by applying a distance measure between the internal representation of the two stimuli. 

This principal is used in the paper to define the "detectability" of the difference between internal representations. The detectability is given as follows:

<img src="https://render.githubusercontent.com/render/math?math=\Large D(x,\varepsilon)%20=%20c_2%20%20\sum_i%20\left|\left|\frac{|\varepsilon_i|^2%20\ast%20h_i}{|x_i|^2\ast%20h_s%20%2B%20c_1}\right|\right|_1.">

Here:
* ![x](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+x): denotes the clean audio stimuli.

