# particle_sizing
Particle Sizing Method to accompany OpenPTV software presented in  [[1]](#1).

This repository contains: 

-  the ptv folder with the data "multiplane" 
-  the ptv folder with data for water droplets "drops"
-  one python file "sizing_calibartion.py" which shows how to use this code to obtain parameters for size correction
-  one python file "utils.py" with the created functions 
-  three python notebooks, with the data extraction to calculate the sizes in the cases: G2 (array of plastic beads) and waters droplets

The data folder can be handled as a standard OpenPTV data folder, with the only difference it contains multiple results folders to differentiate the results of different objects. The folders are described as follows:

- res_0: contains results for flat circles in depth z=0
- res_m4: contains results for flat circles in depth z=-4cm
- res_m8: contains results for flat circles in depth z=-8cm
- res_p4: contains results for flat circles in depth z=+4cm
- res_p8: contains results for flat circles in depth z=+8cm
- res_part: contains results for G1
- res_beads: contains results for G2

## How to use the data

First the "sizing_calibartion.py" should be used to obtain the parameters for corrections as presented in the paper[[1]](#1).

The notebooks contain the information necessary to plot the results after obtaining parameters with "sizing_calibartion.py". "G2_array_results.ipynb" contains an example on how to use the utils functions to estimate sizes on static particles. "Droplets_results.ipynb" contains an example on how to use the utils functions to estimate sizes on a time series.

A new magnification adjustment can be made through the file "sizing_calibartion.py", where the magnification is calculated by the use of the images for multiple plane calibration, the polinomial parameters are save in a txt file and used in the notebooks.

## References
<a id="1">[1]</a> 
Ramirez de la Torre, R. G., and Atle Jensen. 
"A method to estimate the size of particles using the open source software OpenPTV." 
arXiv e-prints (2022): arXiv-2203.
