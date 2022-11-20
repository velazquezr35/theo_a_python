# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:58:07 2022

@author: ramon
"""

import matplotlib.pyplot as plt
import math_tools as utis
import numpy as np


def directors_plot(ax, prof, scale = 0.05):
    for loc_beta, loc_x, loc_y in zip(prof.betas, prof.x_points[:-1], prof.y_points[:-1]):
        p0, pt, pn = utis.gen_norm(loc_beta, [loc_x, loc_y], scale)
        ax.plot([p0[0],pt[0]], [p0[1], pt[1]], 'r')
        ax.plot([p0[0],pn[0]], [p0[1], pn[1]], 'b')
            
def plot_tester(prof):
    fig, ax = plt.subplots()
    ax.plot(prof.x_points, prof.y_points)
    # ax.plot(prof.x_coords, prof.y_coords, 'ro')
    # ax.plot(prof.x_mid, prof.y_mid, 'bo')
    # directors_plot(ax, prof)
    ax.grid()
    ax.set_aspect('equal')
    return fig,ax

def test_PL_point(ax, prof, Ploc, i):
    dir_beta = prof.betas[i]
    r0 = np.array([prof.x_points[i], prof.y_points[i]])
    r_g = utis.RLOC_GLOB(Ploc, r0, dir_beta)
    ax.plot(r_g[0], r_g[1], 'rs')
    
def plot_field(ax, X,Y,U,W, **kwargs):
    if 'density' in kwargs:
        density = kwargs.get('density')
    else:
        density = 1
    # fig, ax = plt.subplots(dpi=100)
    ax.streamplot(X, Y, U, W, density = density)
    # return fig, ax
    
def plot_Vmap(ax, X, Y, U, W, **kwargs):
    Z = np.sqrt(U**2+W**2)
    ax.imshow(Z)
    
def plot_CPs(prof, CPs, **kwargs):
    fig, ax = plt.subplots()
    ax.plot(prof.x_mid, CPs)
    return fig, ax

def plot_dLift(prof, dLift, **kwargs):
    fig, ax = plt.subplots()
    ax.plot(prof.x_mid, dLift)
    return fig, ax

def plot_gammas(prof, gamma_vect, **kwargs):
    fig, ax = plt.subplots()
    ax.plot(prof.x_points, gamma_vect)
    return fig, ax