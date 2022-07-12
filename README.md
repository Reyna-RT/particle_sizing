# particle_sizing
Particle Sizing Method to accompany OpenPTV software

This repository contains: 

-  the ptv folder with the data "multiplane" 
-  one python file "sizing_calibartion.py" which
-  one python file "utils.py" with the created functions 
-  three python notebooks, with the data extraction to calculate the sizes in the three different cases: flat circles, lead weights and glass beads

The data folder can be handled as a standard OpenPTV data folder, with the only difference it contains multiple results folders to differentaite the "rt_is" and "ptv_is" files. The folders are described as follows:

- res_0: contains results for flat circles in depth z=0
- res_m4: contains results for flat circles in depth z=-4cm
- res_m8: contains results for flat circles in depth z=-8cm
- res_p4: contains results for flat circles in depth z=+4cm
- res_p8: contains results for flat circles in depth z=+8cm
- res_part: contains results for lead weights (30 particles in total)
- res_beads: contains results for glass beads (57 particles in total)

## How to use the data

The notebooks contain the information necessary to plot the results of the different tests, "Confirmation_positions_sizes-Circles" contains also the development of the intensity adjustment in th esection "no corrections at all". 

A new magnification adjustment can be made through the file "sizing_calibartion.py", where the magnification is calculated by the use of the images for multiple plane calibration, the polinomial parameters are save in a txt file and used in the notebooks in the "Scale" section of the code.
