# -*- coding: utf-8 -*-
"""
VOR2D func

Nov 2022
"""
import numpy as np
import math_tools as utis    

def VOR2D_L(gamma_1, gamma_2, X_i, x2, x1 = 0):
    gamma_1, gamma_2 = utis.checkarr(gamma_1, gamma_2)
    if abs(X_i[1]) < 1e-9:
        X_i[1]=0
    
    dx1 = X_i[0]-x1
    dx2 = X_i[0]-x2
    R1 = dx1**2 + X_i[1]**2
    R2 = dx2**2 + X_i[1]**2
    theta_2 = np.arctan2(X_i[1],dx2)
    theta_1 = np.arctan2(X_i[1],dx1)
    dtheta = theta_2 - theta_1

    u1 = gamma_1*dtheta/(2*np.pi)
    u2 = gamma_2*(X_i[1]*np.log(R1/R2) + 2*X_i[0]*dtheta)/(4*np.pi)
    
    w1 = -gamma_1*np.log(R1/R2)/(4*np.pi)
    w2 = - gamma_2*(X_i[0]*0.5*np.log(R1/R2) + (x1-x2) + X_i[1]*dtheta)/(2*np.pi)
    u1,u2,w1,w2 = utis.checkabs(u1,u2,w1,w2)
    return(u1, u2, w1, w2)
    
def EVAL_FIELD(x, y, prof, gamma_vect, Vinf, alone = False):
    U = 0
    W = 0
    for i in range(prof.M):
        r0 = [prof.x_points[i], prof.y_points[i]]
        X_i = utis.RGLOB_LOC(np.array([x,y]), r0, prof.betas[i])
        u1, u2, w1, w2 = VOR2D_L(gamma_vect[i], gamma_vect[i+1], X_i, prof.dL[i], x1 = 0)
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
    for i in range(size):
        for j in range(size):
            U[i,j],W[i,j] = EVAL_FIELD(X[i,j], Y[i,j], prof, gamma_vect, Vinf, alone)
    return(U,W)
            
def GAMMA_PROD(inf_mat, RHS):
    return np.matmul(np.linalg.inv(inf_mat), RHS)
           
def GEN_INF_MATRX(prof, Vinf, sol='VOR2D_L'):
    Vinf = utis.LST_ARR(Vinf)
    if sol == 'VOR2D_L':
        inf_mat = np.zeros((prof.N, prof.N))
        RHS = np.zeros((prof.N,1))
        for i in range(prof.M):
            for j in range(prof.M):
                r0 = [prof.x_points[j], prof.y_points[j]]
                X_i = utis.RGLOB_LOC(np.array([prof.x_mid[i], prof.y_mid[i]]), r0, prof.betas[j])
                u1, u2, w1, w2 = VOR2D_L(1, 1, X_i, prof.dL[j], x1 = 0)
                u1,w1 = utis.rot_vect(utis.MAT_L2G(prof.betas[j]), np.array([u1,w1]))
                u2,w2 = utis.rot_vect(utis.MAT_L2G(prof.betas[j]), np.array([u2,w2]))
                inf_mat[i,j] += np.dot([u1,w1], prof.norms[i])
                inf_mat[i,j+1] += np.dot([u2, w2], prof.norms[i])
            RHS[i] = -np.dot(Vinf, prof.norms[i])
        inf_mat[prof.N-1,0] = 1
        inf_mat[prof.N-1, -1] = -1
        return inf_mat, RHS
        
def CP(prof, gamma_vect, Vinf, sol = 'VOR2D_L'):
    Vinf = utis.LST_ARR(Vinf)
    Q_inf = np.sqrt(Vinf[0]**2 + Vinf[1]**2)
    print(Q_inf)
    if sol == 'VOR2D_L':
        CP = np.zeros(prof.M)
        for i in range(prof.M):
            loc_Qt = 0
            U = 0
            W = 0
            for j in range(prof.M):
                r0 = [prof.x_points[j], prof.y_points[j]]
                X_i = utis.RGLOB_LOC(np.array([prof.x_mid[i], prof.y_mid[i]]), r0, prof.betas[j])
                u1, u2, w1, w2 = VOR2D_L(gamma_vect[j], gamma_vect[j+1], X_i, prof.dL[j], x1 = 0)
                u1,w1 = utis.rot_vect(utis.MAT_L2G(prof.betas[j]), np.array([u1,w1]))
                u2,w2 = utis.rot_vect(utis.MAT_L2G(prof.betas[j]), np.array([u2,w2]))
                U += u1+u2
                W += w1+w2
            loc_Qt = np.dot([U,W], prof.tgs[i])
            CP[i] = 1-(loc_Qt**2/Q_inf**2)
        return CP
    
def CL(prof, gamma_vect, Vinf, ro, sol = 'VOR2D_L'):
    if sol == 'VOR2D_L':
        dLift = np.zeros(prof.M)
        Q_inf = np.sqrt(Vinf[0]**2 + Vinf[1]**2)
        for i in range(prof.M):
            dLift[i] = ro*Q_inf*(gamma_vect[i]+gamma_vect[i+1])*0.5*prof.dL[i]
            
        return dLift, sum(dLift)