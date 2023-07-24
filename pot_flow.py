"""
#Theo_a_python
A simple tool for analysis of 2D airfoils using potential flow theory
Methodology: Linear vorticity distribution
Version: Python prepro, solver & postpro

@author: Rodrigo R. Velazquez

AUG 2023

#Module
Pot_flow

- Potential flow solver
- - Method: Discrete linear vortex distribution
"""

#Imports

import numpy as np
import math_tools as utis

#Funcs

def VOR2D_L(gamma_1, gamma_2, X_i, x2, i, j, mode, x1 = 0):
    gamma_1, gamma_2 = utis.checkarr(gamma_1, gamma_2)

    dx1 = X_i[0]-x1
    dx2 = X_i[0]-x2
    R1 = np.sqrt(dx1**2 + X_i[1]**2)
    R2 = np.sqrt(dx2**2 + X_i[1]**2)
    theta_2 = np.arctan2(X_i[1],dx2)
    theta_1 = np.arctan2(X_i[1],dx1)
    dtheta = theta_2 - theta_1
    
    if mode == 'gen':
        if i == j:
            u1 = -0.5*gamma_1*(X_i[0]-x2)/x2
            u2 = 0.5*gamma_2*X_i[0]/x2
            w1 = -gamma_1/(2*np.pi)
            w2 = gamma_2/(2*np.pi)
        else:
            u1 = -gamma_1*(X_i[1]*np.log(R2/R1)+X_i[0]*dtheta - x2*dtheta)/(2*np.pi*x2)
            u2 = gamma_2*(X_i[1]*np.log(R2/R1) + X_i[0]*dtheta)/(2*np.pi*x2)
                
            w1 = -gamma_1*((x2-X_i[1]*dtheta)-X_i[0]*np.log(R1/R2)+x2*np.log(R1/R2))/(2*np.pi*x2)
            w2 = gamma_2*((x2-X_i[1]*dtheta)-X_i[0]*np.log(R1/R2))/(2*np.pi*x2)
    else:
            u1 = -gamma_1*(X_i[1]*np.log(R2/R1)+X_i[0]*dtheta - x2*dtheta)/(2*np.pi*x2)
            u2 = gamma_2*(X_i[1]*np.log(R2/R1) + X_i[0]*dtheta)/(2*np.pi*x2)
                
            w1 = -gamma_1*((x2-X_i[1]*dtheta)-X_i[0]*np.log(R1/R2)+x2*np.log(R1/R2))/(2*np.pi*x2)
            w2 = gamma_2*((x2-X_i[1]*dtheta)-X_i[0]*np.log(R1/R2))/(2*np.pi*x2)

    return(u1, u2, w1, w2)
    
def EVAL_FIELD(x, y, prof, gamma_vect, Vinf,i,j, alone = False):
    U = 0
    W = 0
    for i in range(prof.M):
        r0 = [prof.x_points[i], prof.y_points[i]]
        X_i = utis.RGLOB_LOC(np.array([x,y]), r0, prof.betas[i])
        u1, u2, w1, w2 = VOR2D_L(gamma_vect[i], gamma_vect[i+1], X_i, prof.dL[i],0,0,0, x1 = 0)
        u1,w1 = utis.rot_vect(utis.MAT_L2G(prof.betas[i]), np.array([u1,w1]))
        u2,w2 = utis.rot_vect(utis.MAT_L2G(prof.betas[i]), np.array([u2,w2]))
        U += u1+u2
        W += w1+w2
    if alone:
        return(U, W)
    else:
        return(U+Vinf[0],W+Vinf[1])

def FLOW_FIELD(X, Y, prof, gamma_vect, Vinf, size, alone):
    U,W = np.zeros((2, size, size))
    Vinf = utis.LST_ARR(Vinf)
    for p in range(size):
        for q in range(size):
            U[p,q],W[p,q] = EVAL_FIELD(X[p,q], Y[p,q], prof, gamma_vect, Vinf,0,0, alone)
    return(U,W)
            
def GAMMA_PROD(inf_mat, RHS):
    return np.matmul(np.linalg.inv(inf_mat), RHS)
           
def GEN_INF_MATRX(prof, Vinf, sol='VOR2D_L'):
    Vinf = utis.LST_ARR(Vinf)
    if sol == 'VOR2D_L':
        inf_mat = np.zeros((prof.N, prof.N))
        tg_mat = np.zeros((prof.N, prof.N))
        RHS = np.zeros((prof.N,1))
        for i in range(prof.M):
            for j in range(prof.M):
                r0 = [prof.x_points[j], prof.y_points[j]]
                X_i = utis.RGLOB_LOC(np.array([prof.x_mid[i], prof.y_mid[i]]), r0, prof.betas[j])
                u1, u2, w1, w2 = VOR2D_L(1, 1, X_i, prof.dL[j], i, j, mode = 'gen', x1 = 0)
                u1,w1 = utis.rot_vect(utis.MAT_L2G(prof.betas[j]), np.array([u1,w1]))
                u2,w2 = utis.rot_vect(utis.MAT_L2G(prof.betas[j]), np.array([u2,w2]))
                inf_mat[i,j] += np.dot([u1,w1], prof.norms[i])
                tg_mat[i,j] += np.dot([u1,w1], prof.tgs[i])
                tg_mat[i, j+1] += np.dot([u2,w2], prof.tgs[i])
                inf_mat[i,j+1] += np.dot([u2, w2], prof.norms[i])
            RHS[i] = -np.dot(Vinf, prof.norms[i])
        inf_mat[prof.N-1,0] = 1
        inf_mat[prof.N-1, -1] = 1
        return inf_mat, RHS, tg_mat
        
def coef_CL(prof, gamma_vect, Vinf, tg_mat):
    CL = 0
    V_induc = np.matmul(tg_mat,gamma_vect)
    for i in range(prof.M):
        CL += (np.dot(V_induc[i]+Vinf, prof.tgs[i]))*prof.dL[i]
    return CL
    
def CPCL(prof, gamma_vect, Vinf, sol = 'VOR2D_L'):
    if sol == 'VOR2D_L':
        dCP = np.zeros(prof.M)
        dLift = np.zeros(prof.M)
        Vmod = np.sqrt(Vinf[0]**2 + Vinf[1]**2)
        for i in range(prof.M):
            loc_V = np.dot(Vinf, prof.tgs[i])/Vmod
            U = 0
            W = 0
            for j in range(prof.M):
                r0 = [prof.x_points[j], prof.y_points[j]]
                X_i = utis.RGLOB_LOC(np.array([prof.x_mid[i], prof.y_mid[i]]), r0, prof.betas[j])
                u1, u2, w1, w2 = VOR2D_L(gamma_vect[j], gamma_vect[j+1], X_i, prof.dL[j],i,j, mode = 'gen', x1 = 0)
                u1, w1 = utis.rot_vect(utis.MAT_L2G(prof.betas[j]), np.array([u1,w1]))
                u2, w2 = utis.rot_vect(utis.MAT_L2G(prof.betas[j]), np.array([u2,w2]))
                U += u1+u2
                W += w1+w2
            loc_V += np.dot([U,W], prof.tgs[i])/Vmod
            dCP[i] = 1-(loc_V**2)
            dLift[i] = 1.225*Vmod*(gamma_vect[i]+gamma_vect[i+1])*0.5*prof.dL[i]
            
        return dCP, dLift