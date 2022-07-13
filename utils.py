#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 20:24 2022

@author: reynar
"""

from skimage import io
from skimage.feature import blob_log
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# Function to find distance to nearest neighbor in the mesh
def scipy_method(data):
    # Distance between the array and itself
    dists = cdist(data, data)
    # Sort by distances
    dists.sort()
    # Select the 1st distance, since the zero distance is always 0.
    # (distance of a point with itself)
    nn_dist = dists[:, 1]
    return nn_dist


# Function to define fit
def func(x, a, b, c):
    return a / (b - c * x)


# Function to obtain distance between holes in planes
def obtain_distance_from_images(name, plot_example=False):
    scales = np.zeros((3))
    cal_image = io.imread(name)
    ## skimage below 0.19.2
    blobs = blob_log(cal_image, min_sigma=10, num_sigma=5, threshold=.06, overlap=0.01)
    ## skimage 0.19.2
    # blobs = blob_log(cal_image, min_sigma=10, num_sigma=5, overlap=0.01, threshold_rel=.08 )
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    data = blobs[:, 0:2]
    dist_mean = scipy_method(data).mean()
    dist_std = scipy_method(data).std()
    scales[0] = dist_mean
    scales[1] = dist_std
    scales[2] = dist_std / np.sqrt(len(data))
    if plot_example:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.imshow(cal_image, 'gray')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            ax.add_patch(c)
        plt.show()
    return scales


# Function to define polynomials for magnification
def define_scale_polynomials(scales_matrix, planes, saving_folder, plot=False):
    if not saving_folder:
        saving_folder = './'
    polynomials = []
    if plot:
        plt.figure()
    for i in range(scales_matrix.shape[2]):
        scale = scales_matrix[:, 0, i]
        stdev = scales_matrix[:, 1, i]
        pol = np.polyfit(planes, scale, 1, w=1 / stdev)
        polynomials.append(pol)
        if plot:
            p = np.poly1d(pol)
            plt.plot(planes, scale, 'C%io' % i, label=r'$C_{%i}$' % (i + 1))
            plt.plot(np.arange(-85, 85, 5), p(np.arange(-85, 85, 5)), 'C%i' % i)
            plt.errorbar(planes, scale, yerr=stdev, fmt='C%io' % i, uplims=True, lolims=True)
    if plot:
        plt.legend()
        plt.ylabel('Magnification')
        plt.xlabel('z [mm]')
        plt.savefig(saving_folder + 'polynomials-cameras.png')
    np.savetxt(saving_folder + 'polynomials_2022.txt', polynomials)
    return polynomials


def define_intensity_polynomials(names, size_r, saving_folder, plot=False):
    data = np.zeros((15, 5, 5))
    for i in range(len(names)):
        X, Y, Z, R, G = np.loadtxt(names[i])
        data[:, 0, i] = X
        data[:, 1, i] = Y
        data[:, 2, i] = Z
        data[:, 3, i] = R
        data[:, 4, i] = G

    I_sum = data[:, 4, :]
    shift = np.empty_like(data[:, 1, :])
    for i in range(5):
        shift[:, i] = data[:, 3, i] / size_r
    arg1 = np.where(shift > 1.08)
    arg2 = np.where(shift < 1.08)
    M = np.max(I_sum)
    pol = np.polyfit(I_sum[arg1].flatten() / M, shift[arg1].flatten(), 2)
    p = np.poly1d(pol)
    pol2 = np.polyfit(I_sum[arg2].flatten() / M, shift[arg2].flatten(), 2)
    p2 = np.poly1d(pol2)

    if plot:
        plt.plot(data[:, 4, :] / M, shift, 'o')
        ex = np.linspace(0.3, 1, 100)
        plt.plot(ex, p(ex))
        plt.plot(ex, p2(ex))
        plt.xlabel('Intensity per pixel')
        plt.ylabel('Ratio estimated vs real size')
        plt.show()
    np.savetxt(saving_folder + 'intensity_2022.txt', [pol, pol2])
    return [pol, pol2]


###first polinomial adjust the sizes from different pixel to world transform dependent on z
def magnification(name):
    polynomials = np.loadtxt(name)
    p = []
    for pol in polynomials:
        p.append(np.poly1d(pol))
    return p

###Second function to adjust for gray level and size in pixels
def resize(x, size, name='data/intensity_2022.txt'):
    pols = np.loadtxt(name)
    if size < 4.8:
        P = np.poly1d(pols[0])
    else:
        P = np.poly1d(pols[1])
    return P(x)


def obtain_intensity_maxima(names, count=0):
    if isinstance(count, tuple):
        all_names = [names + 'cam%i.%i_targets' % (i, 10000 + j) for i in range(1, 5) for j in range(count[0],count[1])]
    elif count == 0:
        all_names = [names + 'cam%i.%i_targets' % (i, 10000) for i in range(1,5)]
    elif count > 0:
        all_names = [names + 'cam%i.%i_targets' % (i, 10000 + j) for i in range(1,5) for j in range(count)]
    intensity = []
    for file_name in all_names:
        data = np.loadtxt(file_name, skiprows= 1, usecols=(3,6))
        intensity.append(np.max(data[:,1]/data[:,0]))
    M = np.max(intensity)
    return M

## Obtainin ID from targets
def obtain_values_from_ID(ID, scale, name_feat='', count=1):
    C = []
    for i in range(len(ID)):
        if ID[i] < 0:
            C.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        else:
            c_name = name_feat + 'cam%i.%i_targets' % (i + 1, 10000 + count)
            C.append(np.loadtxt(c_name, skiprows=ID[i] + 1, usecols=(3, 4, 5, 6), max_rows=1))
    al = np.nanmean([C[0][1] / scale[0], C[1][1] / scale[1], C[2][1] / scale[2], C[3][1] / scale[3]])
    b = np.nanmean([C[0][2] / scale[0], C[1][2] / scale[1], C[2][2] / scale[2], C[3][2] / scale[3]])
    return al, b, np.nanmean([C[0][3] / C[0][0], C[1][3] / C[1][0], C[2][3] / C[2][0], C[3][3] / C[3][0]])


### Obtaining data from PTV and calculate size
def Calculate_D_from_PTV(a, name_feat='', num=1):
    raise notImplemented
