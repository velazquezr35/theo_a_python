# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 19:49:55 2022

@author: Rodrigo Ramón Velazquez
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt
import prof_discr

#Global sets
prof_discr.glob_print = False

#Classes
class aero_prof(prof_discr.NACA4Prof):
    
    def __init__(self, alpha, tipo = 'p', code = '',c = 1, gen_xcoords = None, coefs = None):
        super().__init__(tipo, code, c, gen_xcoords, coefs)
        self.Q = 5
        self.ro = 1.225
        self.glob_alpha = np.radians(alpha)
        self.Vinf = self.Q*np.array([np.cos(self.glob_alpha), np.sin(self.glob_alpha)])
        
    def order_coords(self):
        self.panel_coords = np.zeros((len(self.XL)*2-1,2))
        self.panel_coords[:,0] = np.append(np.flip(self.XL), self.XU[1:])
        self.panel_coords[:,1] = np.append(np.flip(self.YL), self.YU[1:])
        self.panel_coords[-1,:] = np.copy(self.panel_coords[0,:])
        
    def order_colocation(self):
        self.df = np.diff(self.panel_coords,axis=0)
        self.colocation_points = self.panel_coords[:-1] + self.df/2

    def calc_angles(self):
        self.angles = np.arctan2(self.df[:,1],self.df[:,0])
        self.dir_cos = np.cos(self.angles)
        self.dir_sin = np.sin(self.angles)
        
    def final_inf_vel(self,i, j, gamm=1):
        if i ==j:
            up = -0.5
            wp = 0
        else:
            X_g = self.colocation_points[i]
            X1_g = self.panel_coords[j]
            X2_g = self.panel_coords[j+1]
            g_l_rot = np.array([[self.dir_cos[j], self.dir_sin[j]],[-self.dir_sin[j], self.dir_cos[j]]])
            X_l = np.matmul(g_l_rot, X_g-X1_g)
            X1_l = X1_g-X1_g #trivial, may be Ro different one time
            X2_l = np.matmul(g_l_rot, X2_g-X1_g)
            r1 = np.power(X_l[0]-X1_l[0],2) + np.power(X_l[1]-X1_l[1],2)
            r2 = np.power(X_l[0]-X2_l[0],2) + np.power(X_l[1]-X2_l[1],2)
            th2 = np.arctan((X_l[1]-X2_l[1]) / (X_l[0]-X2_l[0]))
            th1 = np.arctan((X_l[1]-X1_l[1]) / (X_l[0]-X1_l[0]))
            up = gamm/(2*np.pi)*(th2-th1)
            wp = -gamm/(4*np.pi)*np.log(r1/r2)
        return(self.rot_global_inf(up, wp, j))
        
    def rot_global_inf(self, up,wp, j):
        l_g_rot = np.array([[self.dir_cos[j], -self.dir_sin[j]],[self.dir_sin[j], self.dir_cos[j]]])
        return(np.matmul(l_g_rot, np.array([up, wp])))


    def gen_matrix(self):
        self.N = len(self.colocation_points)
        self.matt = np.zeros((self.N, self.N))
        for i in range(self.N-1):
            for j in range(self.N):
                self.matt[i,j] = np.dot(self.final_inf_vel(i, j), np.array([self.dir_cos[i], -self.dir_sin[i]]))
        self.matt[-1,0] = 1
        self.matt[-1,-1] = 1
        
    def gen_RHS(self):
        self.RHS = np.dot(self.Vinf,np.array([self.dir_cos, -self.dir_sin]))
        
    def solve(self):
        self.gammes = np.matmul(np.linalg.inv(self.matt), self.RHS)
        
    def CP(self):
        self.cps = 1-np.power((self.Q*np.cos(self.angles+self.glob_alpha)+self.gammes/2)/self.Q,2)
        
    def L(self):
        self.largos = np.sqrt(np.power(self.df[:,0],2)+np.power(self.df[:,1],2))
        self.DL = self.ro*self.Q*self.gammes*self.largos
        self.L = np.sum(self.DL)
        self.CL = self.L/(self.ro*0.5)
        
def plot_prof(aero_prof):
    fig, ax = plt.subplots()
    ax.plot(aero_prof.panel_coords[:,0], aero_prof.panel_coords[:,1])
    ax.plot(aero_prof.colocation_points[-1,0], aero_prof.colocation_points[-1,1], 'ro')
    ax.set_aspect('equal')
    ax.grid()
    return fig,ax

def plot_cps(aero_prof):
    fig, ax = plt.subplots()
    ax.plot(aero_prof.colocation_points[:int(aero_prof.N/2)+1,0], aero_prof.cps[:int(aero_prof.N/2)+1],marker='s')
    ax.plot(aero_prof.colocation_points[int(aero_prof.N/2):,0], aero_prof.cps[int(aero_prof.N/2):], marker = 'o')
    
#Tests / debug

x_coord = 0.5*(1-np.cos(np.linspace(0,np.pi,90)))
ramon = aero_prof(0,'p', '0012', 1, x_coord)

#Override
# ramon.XL = np.array([0, 0.5, 1])
# ramon.XU = np.array([0, 0.5, 1])
# ramon.YL = np.array([0, -0.5, 0])
# ramon.YU = np.array([0, 0.5, 0])

ramon.order_coords()
ramon.order_colocation()
ramon.calc_angles()
ramon.gen_matrix()
ramon.gen_RHS()
ramon.solve()
ramon.CP()
ramon.L()
plot_cps(ramon)
# plot_prof(ramon)