# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 12:34:03 2022

@author: ramon
"""

import os

def quitter_lst(lst):
    new = []
    for loc in lst:
        try:
            new.append(float(loc))
        except:
            pass
    return new

def read_CL_alpha(fileloc, header_init = 12):
    f1 = open(fileloc)
    f1_lines = f1.readlines()[12:]
    f1.close()
    alphas = []
    CLs = []
    for i in range(len(f1_lines)):
        loc_l = f1_lines[i].split(' ')
        loc_l = quitter_lst(loc_l)
        alphas.append(float(loc_l[0]))
        CLs.append(float(loc_l[1]))
    return(alphas, CLs)