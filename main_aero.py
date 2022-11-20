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
    
    def __init__(self, alpha, tipo = 'p', code = '', c = 1, gen_xcoords = None, coefs = None):
        super().__init__(tipo, code, c, gen_xcoords, coefs)
        self.Q = 10
        self.ro = 1.225
        self.glob_alpha = np.radians(alpha)
        self.Vinf = self.Q*np.array([np.cos(self.glob_alpha), np.sin(self.glob_alpha)])
        
    def spatial_discretization(self):
        self.panel_coords = np.zeros((len(self.XL)*2-1,2))
        self.panel_coords[:,0] = np.append(np.flip(self.XL), self.XU[1:])
        self.panel_coords[:,1] = np.append(np.flip(self.YL), self.YU[1:])
        self.panel_coords[-1,:] = np.copy(self.panel_coords[0,:])
        
        self.df = np.diff(self.panel_coords,axis=0)
        self.colocation_points = self.panel_coords[:-1] + self.df*0.5

        self.angles = np.arctan2(self.df[:,1],self.df[:,0])
        self.dir_cos = np.cos(self.angles)
        self.dir_sin = np.sin(self.angles)
        
    
    def inf_vel(self, i, j, gamm=1):
        
        X_g = self.colocation_points[i]
        X1_g = self.panel_coords[j]
        X2_g = self.panel_coords[j+1]
        
        g_l_rot = np.array([[self.dir_cos[j], self.dir_sin[j]],[-self.dir_sin[j], self.dir_cos[j]]])
        
        X_gl = X_g - X1_g
        X1_gl = X1_g - X1_g
        X2_gl = X2_g - X1_g
        
        X_ll = np.matmul(g_l_rot, X_gl)
        X1_ll = np.matmul(g_l_rot, X1_gl)
        X2_ll = np.matmul(g_l_rot, X2_gl)
        
        r1 = np.sqrt(np.power(X_ll[0]-X1_ll[0],2)+np.power(X_ll[1]-X1_ll[1],2))
        r2 = np.sqrt(np.power(X_ll[0]-X2_ll[0],2)+np.power(X_ll[1]-X2_ll[1],2))
        
        theta_1 = np.arctan2(X_ll[1]-X1_ll[1],X_ll[0]-X1_ll[0])
        theta_2 = np.arctan2(X_ll[1]-X2_ll[1],X_ll[0]-X2_ll[0])
        
        up = gamm/(2*np.pi)*(theta_2-theta_1)
        wp = gamm/(2*np.pi)*np.log(r2/r1)
        if i == j:
            up = -0.5
            wp = 0
        l_g_rot = np.array([[self.dir_cos[j], -self.dir_sin[j]],[self.dir_sin[j], self.dir_cos[j]]])
        ug, wg = np.matmul(l_g_rot, np.array([up,wp]))
        return(np.array([ug, wg]))


    def gen_matrix(self):
        self.N = len(self.colocation_points)
        self.matt = np.zeros((self.N, self.N))
        for i in range(self.N-1):
            for j in range(self.N):
                # self.matt[i,j] = np.dot(self.inf_vel(i, j), np.array([-self.dir_sin[i], self.dir_cos[i]]))
                self.matt[i,j] = np.dot(self.inf_vel(i, j), np.array([self.dir_cos[i], self.dir_sin[i]]))
                # if i == j:
                #     self.matt[i,j] = -0.5
        self.matt[-1,0] = 1
        self.matt[-1,-1] = 1
        
    def gen_RHS(self):
        self.RHS = np.dot(-self.Vinf,np.array([self.dir_cos, self.dir_sin]))
        self.RHS[-1] = 0
        
    def solve(self):
        self.gammes = np.matmul(np.linalg.inv(self.matt), self.RHS)
        
    def final_vel(self):
        self.total_vels = np.zeros((self.N, 2))
        self.total_vels += self.Vinf
        self.tg_vels = np.zeros(self.N)
        for i in range(0, self.N):
            for j in range(0,self.N):
                self.total_vels[i] += self.inf_vel(i,j, self.gammes[j])
        self.tg_vels = self.total_vels[:,0]*self.dir_cos + self.total_vels[:,1]*self.dir_sin
        
    def new_cp(self):
        self.new_cp = 1 - np.power(self.tg_vels/self.Q, 2)
                
        
    def CP(self):
        self.cps = 1-np.power((self.Q*np.cos(self.angles+self.glob_alpha)+self.gammes*0.5)/self.Q,2)
        
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

def plot_cps(aero_prof, cond = True):
    fig, ax = plt.subplots()
    scale_factor =  1 #max(aero_prof.tg_vels)
    ax.plot(aero_prof.colocation_points[:int(aero_prof.N/2),0], aero_prof.tg_vels[:int(aero_prof.N/2)]/scale_factor,marker='s', label = 'upper')
    ax.plot(aero_prof.colocation_points[int(aero_prof.N/2):,0], aero_prof.tg_vels[int(aero_prof.N/2):]/scale_factor, marker = 'o', label = 'lower')
    ax.plot(aero_prof.colocation_points[-1,0], aero_prof.tg_vels[-1], 'ro')
    
    if cond:
        ax.plot(aero_prof.panel_coords[:,0], aero_prof.panel_coords[:,1])
        # ax.plot(aero_prof.colocation_points[:,0], aero_prof.colocation_points[:,1])
        # ax.set_aspect('equal')
    ax.legend()

#Tests / debug

# x_coord = 0.5*(1-np.cos(np.linspace(0,np.pi,91)))
# x_coord = np.linspace(0,1,91)
# ramon = aero_prof(10,'p', '0012', 1, x_coord)

# ramon.spatial_discretization()
# ramon.gen_matrix()
# ramon.gen_RHS()
# ramon.solve()
# ramon.CP()
# ramon.L()

# ramon.final_vel()
# ramon.new_cp()
# print(ramon.L, ramon.CL)
# plot_cps(ramon)
# # plot_prof(ramon)