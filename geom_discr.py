# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 17:43:26 2022

@author: ramon
"""

import numpy as np
import math_tools as utis
   

class Prof_gen:
    
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

 
# # Plotting stream plot

# Z = np.sqrt(u**2+w**2)
# ax.imshow(Z, interpolation='bilinear', extent = [x[0], x[-1], y[0], y[-1]])
# ax.streamplot(X, Y, u, w, density = 2)
# ax.plot([0,x2],[0,0], linewidth = 5)