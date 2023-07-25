"""
#Theo_a_python
A simple tool for analysis of 2D airfoils using potential flow theory
Methodology: Linear vorticity distribution
Version: Python prepro, solver & postpro

@author: Rodrigo R. Velazquez

AUG 2023

#Module
Geom_discr

- Geometric tools for the airfoil analysis and external data reading
"""

#Imports

import numpy as np
import math_tools as utis
import os

#Classes

class Prof_gen:
    '''
    Profile object
    
    Current init overloads:
        - from_coords: start the profile from a set of X and Y points. The points should be indicated in CW direction from the trailing edge. DO NOT duplicate the last one.
        
        Example:
            
            original_x_dist = array([0.  , 0.25, 0.5 , 0.75, 1.  ])
            x_points = array([0.  , 0.25, 0.5 , 0.75, 1.  , 0.75, 0.5 , 0.25])
            
    Methods:
        - update(): Calculates some geometric data (normals, tangentials, distances, lenghts...)
    '''
    
    def __init__(self,**kwargs):
        if 'from_coords' in kwargs:
            if kwargs.get('from_coords'):
                self.x_coords = np.array(kwargs.get('x_points'))
                self.y_coords = np.array(kwargs.get('y_points'))
    
    def update(self):
        self._gen_p_points()
        self._calc_thet_midpoints()
        self._panel_length()
        self.N = len(self.x_points)
        self.M = self.N-1
        self._gen_norms()
        
    def _gen_norms(self, **kwargs):
        self.norms = []
        self.tgs = []
        for i in range(self.M):
            self.norms.append(utis.rot_vect(utis.MAT_L2G(self.betas[i]), np.array([0,1])))
            self.tgs.append(utis.rot_vect(utis.MAT_L2G(self.betas[i]), np.array([1,0])))
        self.norms = np.array(self.norms)
        self.tgs = np.array(self.tgs)
        
    def _gen_p_points(self, **kwargs):
        #May include some filter about N points later...
        self.x_points = utis.appnd_last(self.x_coords)
        self.y_points = utis.appnd_last(self.y_coords)
        
    def _panel_length(self, **kwargs):
        self.dL = np.sqrt(self.dx**2 + self.dy**2)
        
        
    def _calc_thet_midpoints(self, **kwargs):
        self.dx = np.diff(self.x_points)
        self.dy = np.diff(self.y_points)
        self.betas = np.arctan2(self.dy, self.dx)
        self.x_mid = 0.5*self.dx + self.x_points[:-1]
        self.y_mid = 0.5*self.dy + self.y_points[:-1]
        
#Funcs

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