"""
#Theo_a_python
A simple tool for analysis of 2D airfoils using potential flow theory
Methodology: Linear vorticity distribution
Version: Python prepro, solver & postpro

@author: Rodrigo R. Velazquez

AUG 2023

#Module
Plotter

- Plotting funcs and general postpro tools. Needs math_tools.
"""

#Imports
import matplotlib.pyplot as plt
import math_tools as utis
import numpy as np


#General postpro funcs
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


#Testing dedicated funcs

def _testing_CLvs(validation,CL_lst, **kwargs):
    '''
    Testing func
    
    Plot external CL data vs alpha against calculated CLs for a testing case
    
    Parameters
    ----------
    [validation] : list
        internal: [alphas,CLs]
    CL_lst: list
        Calculated CLs

    Returns
    -------
    fig, ax - Matplotlib fig and ax objs

    '''
    
    fig, ax = plt.subplots()
    ax.plot(validation[0], validation[1], label = 'XFOIL')
    ax.plot(validation[0], CL_lst, label = 'THEO_A')
    
    ax.set_xlabel('alpha [deg]', fontsize = 12)
    ax.set_ylabel('CL', fontsize = 12)
    fig.suptitle('CL vs alpha')
    ax.grid()
    ax.legend(title='Testing - NACA 0012 - RE 500E3')
    
    return fig, ax

def _testing_CPvs(prof, CPs, validation, **kwargs):
    '''
    Testing func
    
    Plot external CP data vs calculated for a testing case
    
    Parameters
    ----------
    prof : theo_a Profile obj
    CPs : nump.ndarray
    validation : list
        internal: [x,cp]

    Returns
    -------
    fig, ax - Matplotlib fig and ax objs

    '''
    fig, ax = plt.subplots()
    ax. plot(prof.x_mid, CPs, label = 'THEO_A')
    ax.plot(validation[0], validation[1], label = 'XFOIL')
    
    ax.set_xlabel('x coord [adim]', fontsize = 12)
    ax.set_ylabel('-Cp [adim]', fontsize = 12)
    fig.suptitle('Pressure coef. dist.')
    ax.grid()
    ax.legend(title='Testing - NACA 0012 - RE 500E3')
    return fig, ax