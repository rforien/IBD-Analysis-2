#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 19:51:32 2023

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

from . import hetero_sharing as hs

def F(p, nu, alpha):
    return 2 * (alpha / (4 * p))**(nu/2) * sp.kv(nu, np.sqrt(alpha * p))

class Homogeneous(object):
    def __init__(self, dim, genome_length):
        assert type(dim) == int and dim >= 1
        assert genome_length > 0
        self.d = dim
        self.G = genome_length
        self.param_names = ['sigma', 'N']
    
    def prepare(self, positions, length_bins):
        n = np.size(positions, axis = 0)
        distances = np.zeros((n,n))
        for i in range(np.size(positions, axis = 1)):
            x = positions[:,i]
            X1, X2 = np.meshgrid(x, x)
            distances = distances + np.abs(X1 - X2)**2
        distances = np.sqrt(distances)
        self.r = np.mean(distances)
        self.distances = distances / self.r
    
    def start_params(self):
        return np.array([5, 10])
    
    def expected_ibd(self, params, length_bins):
        params = dict(zip(self.param_names, params))
        alpha = self.distances**2 / params['sigma']**2
        p = 2 * self.r**2 * length_bins
        a = 1 / (np.abs(params['N']) * (4 * np.pi * params['sigma']**2)**(self.d / 2))
        E_geq_L = a * (F(p[:, np.newaxis, np.newaxis], 
                         1 - self.d / 2, 
                         alpha[np.newaxis,:,:]) 
                       + 2 * (self.G - length_bins[:, np.newaxis, np.newaxis]) * self.r**2
                       * F(p[:, np.newaxis, np.newaxis],
                           2 - self.d / 2,
                           alpha[np.newaxis,:,:]))
        E_ibd = np.concatenate((-np.diff(E_geq_L, axis = 0), E_geq_L[-1,np.newaxis,:,:]),
                               axis = 0)
        return E_ibd

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    
class Heterogeneous(Homogeneous):
    def __init__(self, dim, genome_length, infer_location = False):
        super().__init__(dim, genome_length)
        self.param_names = ['sigma+', 'sigma-', 'N+', 'N-']
        self.infer_location = bool(infer_location)
        if self.infer_location:
            self.param_names = self.param_names + ['x0', 'theta']
    
    def start_params(self):
        params = np.array([0.5, 0.5, 50, 50])
        if self.infer_location:
            params = np.concatenate((params, [np.mean(self.coords[:,0]) + 0.5, np.pi / 4]))
        return params
    
    def prepare(self, positions, length_bins):
        self.coords = positions
    
    def expected_ibd(self, params, length_bins):
        params = dict(zip(self.param_names, params))
        # print(params)
        sigma = np.array([params['sigma-'], params['sigma+']])
        pop_size = np.array([params['N-'], params['N+']])
        # shift coordinates
        if self.infer_location:
            coords = self.coords - np.array([[params['x0'], 0]])
            coords = coords @ rotation_matrix(params['theta']).T
        else:
            coords = self.coords
        step, L = hs.grid_fit(self.coords, sigma = sigma, coarse=0.05)
        L = L + (L % 2)
        bc = hs.barycentric_coordinates(self.coords, L, step)
        E_ibd_cumul = hs.ibd_sharing(bc, L, step, length_bins, self.G, sigma, pop_size,
                                     cumul = True, balance = 'symmetric')
        E_ibd = np.concatenate((-np.diff(E_ibd_cumul, axis = 0), E_ibd_cumul[-1,np.newaxis,:,:]),
                               axis = 0)
        return E_ibd
