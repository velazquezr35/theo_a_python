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
    
    def __init__(self, code):
        super().__init__(code)
        self.Vinf = np.array([1,0])     
        
    def order_coords(self):
        self.panel_coords = np.zeros((len(self.XL)*2-1,2))
        self.panel_coords[:,0] = np.append(np.flip(self.XL), self.XU[1:])
        self.panel_coords[:,1] = np.append(np.flip(self.YL), self.YU[1:])
        self.panel_coords[-1,:] = self.panel_coords[0,:]
        
    def order_colocation(self):
        self.colocation_points = self.panel_coords[:-1] + np.diff(self.panel_coords, axis = 0)*0.5

    def calc_angles(self):
        self.df = np.diff(self.panel_coords,axis=0)
        self.angles = np.arctan2(self.df[:,1],self.df[:,0])
        
    def gen_matrix(self):
        self.N = len(self.colocation_points)
        self.matt = np.zeros((self.N, self.N))
        for i in range(self.N-1):
            for j in range(self.N):
                if i == j:
                    self.matt[i,j] = -0.5
                else:
                    self.matt[i,j] = np.dot(self.inf_vel(i,j), (np.cos(self.angles[j]), -np.sin(self.angles[j])))
        self.matt[-1,0] = 1
        self.matt[-1,-1] = 1
        
    def gen_RHS(self):
        self.RHS = np.dot(-self.Vinf,(np.cos(self.angles), -np.sin(self.angles)))
        self.RHS[-1] = 0
        
    def solve(self):
        self.gammes = np.matmul(np.linalg.inv(self.matt), self.RHS)
        
    def inf_vel(self, i, j, gamm = 1):
        alpha = self.angles[j]
        X = self.colocation_points[i]
        X1 = self.panel_coords[j]
        X2 = self.panel_coords[j+1]
        
        loc_cos = np.cos(alpha)
        loc_sin = np.sin(alpha)
        rot_m = np.array([[loc_cos, -loc_sin],[loc_sin, loc_cos]])
        X1 = np.matmul(rot_m, X1)
        X2 = np.matmul(rot_m, X2)
        print(i,j)
        X = np.matmul(rot_m, X)
        up = gamm/(2*np.pi)*(np.arctan((X[1]-X2[1])/(X[0]-X2[0]))-np.arctan((X[1]-X1[1])/(X[0]-X1[0])))
        wp = gamm/(4*np.pi)*np.log((np.power(X[0]-X2[0],2)+np.power(X[1]-X2[1],2))/(np.power(X[0]-X1[0],2)+np.power(X[1]-X1[1],2)))
            
        ug, wg = np.matmul(np.array([[loc_cos, loc_sin],[-loc_sin, loc_cos]]), [up,wp])
        return(ug, wg)

    def CP(self):
        self.ro = 1.225
        self.Q = self.Vinf[0]
        self.cps = 1-np.power((self.Q*np.cos(self.angles)+self.gammes/2)/self.Q,2)
        
    def L(self):
        self.largos = np.sqrt(np.power(self.df[:,0],2)+np.power(self.df[:,1],2))
        self.DL = 1.225*self.Q*np.array(self.gammes)[0]*self.largos
        self.L = np.sum(self.DL)
        
def plot_prof(aero_prof):
    fig, ax = plt.subplots()
    ax.plot(aero_prof.panel_coords[:,0], aero_prof.panel_coords[:,1])
    ax.plot(aero_prof.colocation_points[-1,0], aero_prof.colocation_points[-1,1], 'ro')
    # ax.plot(aero_prof.colocation_points[:int(aero_prof.N/2 + 1),0], aero_prof.cps[:int(aero_prof.N/2 + 1)])
    ax.set_aspect('equal')
    return fig,ax
    
    
#Tests / debug
ramon = aero_prof('4412')
x_coord = np.linspace(0,1,3)
ramon.y_t(x_coord)
ramon.y_c(x_coord)
ramon.calc_dyc(x_coord)
ramon.calc_theta()
ramon.calc_pos(x_coord)

#Override
ramon.XL = np.array([0, 0.2, 1])
ramon.XU = np.array([0, 0.2, 1])
ramon.YL = np.array([0, -0.1, 0])
ramon.YU = -ramon.YL

ramon.order_coords()
ramon.order_colocation()
ramon.calc_angles()
ramon.gen_matrix()
ramon.gen_RHS()
ramon.solve()
ramon.CP()
ramon.L()
plot_prof(ramon)