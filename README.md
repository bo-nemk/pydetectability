# Implementation of Taal distortion detectability
Implementation is based on the original paper "A Low-complexity spectro-temporal based perceptual model" and the code provided on https://ceestaal.nl/code/.

## Model Implementation
The model by Taal et al. proposes a "perceptual distance measure" to quantify the perceived difference between a clean and degraded audio stimuli. This perceptual distance is given by applying a distance measure between the internal representation of the two stimuli. 

This principal is used in the paper to define the "detectability" of the difference between internal representations. The detectability is given as follows:

![D](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle++D%28x%2C%5Cvarepsilon%29+%3D+c_2+%5Csum_i+%5Cleft%7C%5Cleft%7C%5Cfrac%7B%7C%5Cvarepsilon_i%7C%5E2+%5Cast+h_i%7D%7B%7Cx_i%7C%5E2%5Cast+h_s+%2B+c_1%7D%5Cright%7C%5Cright%7C_1.)

Here:
* ![x](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+x): denotes the clean audio stimuli.
* ![x-i](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+x_i): denotes the difference between the degraded and clean audio stimuli filtered by the ith auditory filter ![h_i](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+h_i)
* ![\varepsilon](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cvarepsilon): denotes the difference between the degraded and clean audio stimuli.
* ![\varepsilon-i](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cvarepsilon_i): denotes the difference between the degraded and clean audio stimuli filtered by the ith auditory filter ![h_i](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+h_i) 
