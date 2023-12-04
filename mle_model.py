#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:54:23 2023

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy.stats as st
import copy

from statsmodels.base.model import GenericLikelihoodModel

'''
def bin_counts(sample_locations, IBD_segments, length_bins, bin_by_location = True):
    if bin_by_location:
        locations, inverse = np.unique(sample_locations.values, axis = 0,
                                       return_inverse=True)
        n = len(locations)
    else:
        n = len(sample_locations)
        inverse = np.arange(n)
    k = len(length_bins)
    counts = np.zeros((n, n, k))
    for (i1, ind1) in enumerate(sample_locations.index):
        for (i2, ind2) in enumerate(sample_locations.index):
            if i1 >= i2:
                continue
            ibd = IBD_segments.loc[np.logical_and(np.isin(IBD_segments['individual1'], 
                                                          [ind1, ind2]),
                                                  np.isin(IBD_segments['individual2'], 
                                                          [ind1, ind2]))]
            lengths = ibd['length'].values
            C = np.sum(lengths[:,np.newaxis] > length_bins[np.newaxis, :], axis = 0)
            bin_counts = np.hstack((-np.diff(C), [C[-1]]))
            ii1 = np.minimum(inverse[i1], inverse[i2])
            ii2 = np.maximum(inverse[i1], inverse[i2])
            counts[ii1, ii2, :] += bin_counts
            # counts[inverse[i2], inverse[i1], :] += bin_counts
    return counts
'''

class MLE_model(GenericLikelihoodModel):
    def __init__(self, IBD_segments, sample_locations, length_bins, model):
        try:
            self.samples = sample_locations.index
        except:
            self.samples = np.arange(np.size(sample_locations, axis = 0))
        self.locations, self.inverse, self.counts = np.unique(sample_locations, 
                                                              axis = 0,
                                                              return_counts = True,
                                                              return_inverse = True)
        self.length_bins = np.array(length_bins)
        assert np.min(np.diff(length_bins)) > 0
        self.K = np.size(self.length_bins)
        self.n = np.size(self.locations, axis = 0)
        counts = self.bin_counts(IBD_segments)
        self.count_pairs() # mask that will be applied to expected ibd sharing
        super().__init__(counts)
        self.n_samples = np.size(sample_locations, axis = 0)
        self.model = model
        self.model.prepare(self.locations, self.length_bins)
        self.full_IBD = IBD_segments
        self.full_locations = sample_locations
    
    def bin_counts(self, IBD_segments):
        print("Binning segments counts...")
        counts = np.zeros((self.K,self.n,self.n))
        for (i1, ind1) in enumerate(self.samples):
            for (i2, ind2) in enumerate(self.samples):
                if i1 >= i2:
                    continue
                ibd = IBD_segments.loc[np.logical_and(np.isin(IBD_segments['individual1'], 
                                                              [ind1, ind2]),
                                                      np.isin(IBD_segments['individual2'], 
                                                              [ind1, ind2]))]
                lengths = ibd['length'].values
                C = np.sum(lengths[:,np.newaxis] >= self.length_bins[np.newaxis, :], axis = 0)
                bin_counts = np.hstack((-np.diff(C), [C[-1]]))
                ii1 = np.minimum(self.inverse[i1], self.inverse[i2])
                ii2 = np.maximum(self.inverse[i1], self.inverse[i2])
                # print("binned counts between samples %d and %d, at coordinates %d, %d." % (ind1, ind2, ii1, ii2))
                # print(bin_counts)
                counts[:, ii1, ii2] = counts[:, ii1, ii2] + bin_counts
        print("Done.")
        return counts
    
    def count_pairs(self):
        N1, N2 = np.meshgrid(np.arange(self.n), np.arange(self.n))
        mask = (N2 <= N1).astype(float)
        C1, C2 = np.meshgrid(self.counts, self.counts)
        self.pairs = C1 * C2
        np.fill_diagonal(self.pairs, self.counts * (self.counts-1) / 2)
        self.pairs = self.pairs * mask
        
    def loglikeobs(self, params):
        expected_ibd = self.pairs[np.newaxis,:,:] * self.model.expected_ibd(params,
                                                                            self.length_bins)
        IBD_counts = self.endog
        clogl = np.nansum(IBD_counts * np.log(expected_ibd) - expected_ibd)# - sp.gammaln(IBD_counts))
        return clogl
    
    def fit(self, *args, **kwargs):
        return super().fit(self.model.start_params(), *args, **kwargs)
    
    def estimate_GodambeIM(self, params, n_bootstrap = 200, n_samples = 100):
        assert type(n_bootstrap) is int and n_bootstrap > 1
        assert type(n_samples) is int and n_samples > 1
        n = np.sum(self.pairs)
        nb_p = len(params)
        J = np.zeros((nb_p, nb_p))
        for i in range(n_bootstrap):
            bs_model = self.bootstrap_model(n_samples)
            U = bs_model.score(params)
            J = J + U[:, np.newaxis] @ U[np.newaxis, :] * n / (n_bootstrap * np.sum(bs_model.pairs))
        H = self.hessian(params)
        G = H @ np.linalg.solve(J, H)
        return G
    
    def bootstrap_model(self, n_samples):
        samples = np.random.choice(self.samples, size = n_samples, replace = False)
        bs_segments = self.full_IBD.loc[samples]
        bs_locations = self.full_locations.loc[samples]
        bs_model = MLE_model(bs_segments, bs_locations, self.length_bins, 
                             copy.copy(self.model))
        return bs_model
    
    def corrected_std(self, params):
        G = self.estimate_GodambeIM(params)
        G_inv = np.linalg.inv(G)
        return np.sqrt(np.diagonal(G_inv))
    
    def compare_expectations(self, params, filter_same_locations = False):
        side = (self.locations[:,0] > 0).astype(float) - (self.locations[:,0] < 0).astype(float)
        S1, S2 = np.meshgrid(side, side)
        both_left = (S1 == -1) * (S2 == -1)
        both_right = (S1 == 1) * (S2 == 1)
        different_sides = (S1 == 1) * (S2 == -1) + (S1 == -1) * (S2 == 1)
        if filter_same_locations:
            same_location = np.eye(np.size(side)).astype(bool)
            masks = [both_left * (~same_location), 
                     both_left * same_location,
                     both_right * (~same_location), 
                     both_right * same_location,
                     different_sides]
            names = ['Both samples from the left - different locations', 
                     'Both samples from the left - same location',
                     'Both samples from the right - different locations',
                     'Both samples from the right - same location',
                     'Samples from different sides']
            colors = ['blue', 'red', 'orange', 'pink', 'green']
        else:
            masks = [both_left, both_right, different_sides]
            names = ['Both samples from the left',
                     'Both samples from the right',
                     'Samples from different sides']
            colors = ['#DD0000', '#0000FF', 'green']
        
        expected_ibd = self.pairs[np.newaxis, :, :] * self.model.expected_ibd(params,
                                                                               self.length_bins)
        IBD_counts = self.endog
        
        plt.figure(figsize = (9,5))
        ax = plt.axes()
        bin_width = np.min(np.diff(self.length_bins))
        n_bars = len(masks)
        bar_width = 0.9 * bin_width / n_bars
        for (i, mask, name, color) in zip(np.arange(n_bars), masks, names, colors):
            offset = (0.5 + i) * bar_width
            ibd_sum = np.sum(IBD_counts[:,mask], axis = 1)
            exp_sum = np.sum(expected_ibd[:,mask], axis = 1)
            ax.bar(self.length_bins[:-1] + offset, ibd_sum[:-1], #color = color, 
                   label = name, width = bar_width)
            errors = np.vstack((exp_sum[:-1] - st.poisson.ppf(0.025, exp_sum[:-1]),
                                st.poisson.isf(0.025, exp_sum[:-1]) - exp_sum[:-1]))
            errbar = ax.errorbar(self.length_bins[:-1] + offset, exp_sum[:-1],
                                yerr = errors,
                                capsize = 3, color = 'black', fmt = '+')
            # ax.hlines(exp_sum[:-1], self.length_bins[:-1] + offset - bar_width / 2,
            #           self.length_bins[:-1] + offset + bar_width / 2, color = 'black',
            #           linewidth = 1.8)
        errbar.set_label('Expected value and 95% confidence interval')
        ax.set_xticks(self.length_bins)
        ax.set_xticklabels(np.round(100 * self.length_bins).astype(int))
        # ax.hlines(exp_sum[0], self.length_bins[0], self.length_bins[0], 
        #           color = 'black', linewidth = 1.8, label = 'expected value')
        ax.legend(loc = 'best')
        # ax.grid(True)
        ax.set_yscale('log')
        ax.set_xlabel('Segment length intervals (cM)')
        ax.set_ylabel('Number of IBD segments')
        return ax
    