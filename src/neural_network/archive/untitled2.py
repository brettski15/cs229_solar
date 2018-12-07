# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:49:14 2018

@author: Eddie
"""

import pandas as pd
import numpy as np

#hyps = pd.read_csv('hyperparameter_search.csv')
#print (hyps)
#hyps = np.array(hyps)
#
#num_layers = hyps[0, :]
#
#layer_dims = []
#dropout = []
#l2_reg = []
#
#for i in range(hyps.shape[1]):
#    layer_dims.append([])
#    dropout.append([])
#    l2_reg.append([])    
#    
#for i in range(hyps.shape[1]):
#    layer_dims[i] = hyps[1:7, i]
#    dropout[i] = hyps[7:13, i]
#    l2_reg[i] = hyps[13:, i]
#    
#print (num_layers, layer_dims, dropout, l2_reg)

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x_mask = x[x < 3]

print (x_mask)