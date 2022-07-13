#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:18:53 2021

@author: R.G. Ramirez de la Torre
Department of Mathematics at the University of Oslo
"""

import numpy as np
from utils import *

###### Defining folders and paths
folder = 'multiplane/'
saving_folder = 'data/'
cal_folder = folder+'cal/'
img_folder = folder+'img/'
res_folder = folder+'res/'
poly_file = 'data/polynomials_2022.txt'
# Number of cameras used
n_cams = 4
# calibration plane positions in millimeters
planes = np.array([-80, -40, 0, 40, 80])
# labels for filenames for each calibration plane
plane_names = ['m8', 'm4', '0', 'p4', 'p8']
file_names = ['m8','m4','zero','p4','p8']
plane = ['m8/','m4/','0/','p4/','p8/']
scales_matrix = np.zeros((len(planes),3,n_cams))
######

### Measured data of position of circles
xr = np.asarray([-30.1,-43,-81,-101,-34.0,-58.5,-87.0,-114.0,-142.5,-169.5,
                  -195.0,-220.0,-245.0,-12.0,-43.0])+4
yr = np.asarray([46.0,45.0,42.0,43.5,85.5,85.0,85.0,85.0,85.0,85.0,85.0,85.5,
                 85.0,147.0,149.0])-15.
zr = np.zeros(15)-80.0
#### Matrix of real diameters
size_r = np.asarray([3.16,3.16,5.65,5.65,0.5,1,1.5,2.5,3,4,5,6,8,10,15])

#### Calculate polynomials for magnification in depth
for pos in range(len(planes)):
    print('Plane z=%i'%planes[pos])
    for cam in range(1, n_cams+1):
        cal_image_name = cal_folder+'cal_'+plane_names[pos]+'_cam'+str(cam)+'.tif'
        # Show image of one of the calibration targets
        if pos == 1 and cam == 3 :
            scales_matrix[pos, :, cam-1] = obtain_distance_from_images(cal_image_name)
        scales_matrix[pos, :, cam-1] = obtain_distance_from_images(cal_image_name)
        print('Done, camera %i'%cam)
        print('10 mm = %.4f, error = %.4f'%(scales_matrix[pos, 0, cam-1], scales_matrix[pos, 2, cam-1]))

define_scale_polynomials(scales_matrix, planes, saving_folder)
# ####

##### Calculate polynomials for intensity change
p = magnification(poly_file)
for j in range(len(planes)):
    # Data storage, loading particle positions and filenames
    folder1 = folder + 'res_' + plane[j]
    name = folder1 + 'rt_is.10001'
    file_name = img_folder + 'circles_' + plane_names[j] + '_'
    a = np.loadtxt(name, skiprows=1)
    d_E = np.zeros(len(a))
    Intens = np.zeros(len(a))
    # Separate data into coordinates, add offset  between ptv and real positions(systematic error)
    x = a[:, 1]
    y = a[:, 2]
    z = a[:, 3]
    # Organize ptv position with measured positions
    part_id = []
    i = 0
    for x0, y0, z0 in zip(xr, yr, zr):
        length = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
        args = np.argmin(length)
        part_id.append(args)
        i += 1
    #obtain diameter approximation from ptv and z polynomial
    for i in range(len(a)):
        # Scale pixel to world with z position
        scale = [p[j](a[i][3]) * 0.1 for j in range(4)]
        # Find ID in each camera
        ID = np.array([int(a[i][4]), int(a[i][5]), int(a[i][6]), int(a[i][7])], dtype=int)
        # Use ID values to obtain axis lengths and intensity
        al, b, gray = obtain_values_from_ID(ID, scale, file_name)
        Intens[i] = gray
        d_E[i] = np.sqrt(al * b)
    # Save data in files

    np.savetxt('no-gray_results_' + plane[j][:-1] + '.txt', (x[part_id], y[part_id], z[part_id],
    d_E[part_id], Intens[part_id]))
    zr = zr + 40.0
# Grab data and find resize polynomials
names = ['no-gray_results_p8.txt','no-gray_results_p4.txt','no-gray_results_0.txt',
         'no-gray_results_m4.txt','no-gray_results_m8.txt']
intensity_pols = define_intensity_polynomials(names, size_r, saving_folder)