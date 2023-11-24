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
        if R1 == 0: #CHECK THIS. OR 0?
            u1 = -0.5*gamma_1*(X_i[0]-x2)/x2
            u2 = 0.5*gamma_2*X_i[0]/x2
            w1 = -gamma_1/(2*np.pi)
            w2 = gamma_2/(2*np.pi)
        else:
            u1 = -gamma_1*(X_i[1]*np.log(R2/R1)+X_i[0]*dtheta - x2*dtheta)/(2*np.pi*x2)
            u2 = gamma_2*(X_i[1]*np.log(R2/R1) + X_i[0]*dtheta)/(2*np.pi*x2)
                
            w1 = -gamma_1*((x2-X_i[1]*dtheta)-X_i[0]*np.log(R1/R2)+x2*np.log(R1/R2))/(2*np.pi*x2)
            w2 = gamma_2*((x2-X_i[1]*dtheta)-X_i[0]*np.log(R1/R2))/(2*np.pi*x2)

    return(u1, u2, w1, w2)
    
def EVAL_FIELD(x, y, wrapper, gamma_vect, Vinf,i,j, alone = False):
    U = 0
    W = 0

    twin_j = 0
    loc_counter_j = 0
    loc_flag_j = wrapper.Ms[loc_counter_j]
    for j in range(wrapper.M_total+wrapper.profile_count-1):
        if not j == loc_flag_j:
            r0 = [wrapper.x_points[j], wrapper.y_points[j]]
            X_i = utis.RGLOB_LOC(np.array([x,y]), r0, wrapper.betas[twin_j])
            u1, u2, w1, w2 = VOR2D_L(gamma_vect[j], gamma_vect[j+1], X_i, wrapper.dL[twin_j],0,0,0, x1 = 0)
            u1, w1 = utis.rot_vect(utis.MAT_L2G(wrapper.betas[twin_j]), np.array([u1,w1]))
            u2, w2 = utis.rot_vect(utis.MAT_L2G(wrapper.betas[twin_j]), np.array([u2,w2]))
            U += u1+u2
            W += w1+w2
            twin_j += 1
        else:
            loc_counter_j += 1
            loc_flag_j += wrapper.Ms[loc_counter_j] + 1
    if alone:
        return(U, W)
    else:
        return(U+Vinf[0],W+Vinf[1])

def FLOW_FIELD(X, Y, wrapper, gamma_vect, Vinf, size, alone):
    U,W = np.zeros((2, size, size))
    Vinf = utis.LST_ARR(Vinf)
    for p in range(size):
        for q in range(size):
            U[p,q],W[p,q] = EVAL_FIELD(X[p,q], Y[p,q], wrapper, gamma_vect, Vinf,0,0, alone)
    return(U,W)
            
def GAMMA_PROD(inf_mat, RHS):
    return np.matmul(np.linalg.inv(inf_mat), RHS)
           
def GEN_INF_MATRX(wrapper, Vinf, sol='VOR2D_L'):
    Vinf = utis.LST_ARR(Vinf)
    if sol == 'VOR2D_L':
        inf_mat = np.zeros((wrapper.N_total, wrapper.N_total))
        tg_mat = np.zeros((wrapper.N_total, wrapper.N_total))
        RHS = np.zeros((wrapper.N_total,1))
        loc_counter_i = 0
        loc_counter_j = 0
        loc_flag_i = wrapper.Ms[loc_counter_i]
        loc_flag_j = wrapper.Ms[loc_counter_j]
        twin_i = 0
        for i in range(wrapper.M_total+wrapper.profile_count-1):
            if not i == loc_flag_i:
                twin_j = 0
                loc_counter_j = 0
                loc_flag_j = wrapper.Ms[loc_counter_j]
                for j in range(wrapper.M_total+wrapper.profile_count-1):
                    if not j == loc_flag_j:
                        r0 = [wrapper.x_points[j], wrapper.y_points[j]]
                        X_i = utis.RGLOB_LOC(np.array([wrapper.x_mid[twin_i], wrapper.y_mid[twin_i]]), r0, wrapper.betas[twin_j])
                        u1, u2, w1, w2 = VOR2D_L(1, 1, X_i, wrapper.dL[twin_j], i, j, mode = 'gen', x1 = 0)
                        u1,w1 = utis.rot_vect(utis.MAT_L2G(wrapper.betas[twin_j]), np.array([u1,w1]))
                        u2,w2 = utis.rot_vect(utis.MAT_L2G(wrapper.betas[twin_j]), np.array([u2,w2]))
                        inf_mat[i,j] += np.dot([u1,w1], wrapper.norms[twin_i])
                        tg_mat[i,j] += np.dot([u1,w1], wrapper.tgs[twin_i])
                        tg_mat[i, j+1] += np.dot([u2,w2], wrapper.tgs[twin_i])
                        inf_mat[i,j+1] += np.dot([u2, w2], wrapper.norms[twin_i])
                        twin_j += 1
                    else:
                        loc_counter_j += 1
                        loc_flag_j += wrapper.Ms[loc_counter_j] + 1
                RHS[i] = -np.dot(Vinf, wrapper.norms[twin_i])
                twin_i += 1
            else:
                loc_counter_i += 1
                loc_flag_i += wrapper.Ms[loc_counter_i] + 1 
        i_counter = wrapper.Ms[0]
        j_counter = 0
        for i in range(0, wrapper.profile_count):
            inf_mat[i_counter, j_counter] = 1
            inf_mat[i_counter, j_counter+wrapper.Ms[i]] = 1
            j_counter += wrapper.Ms[i]+1
            i_counter += wrapper.Ms[i]+1

        return inf_mat, RHS, tg_mat
    
def CPCL(wrapper, gamma_vect, Vinf, sol = 'VOR2D_L'):
    if sol == 'VOR2D_L':
        dCP = np.zeros(wrapper.M_total)
        dLift = np.zeros(wrapper.M_total)
        Vmod = np.sqrt(Vinf[0]**2 + Vinf[1]**2)
        loc_counter_i = 0
        loc_flag_i = wrapper.Ms[loc_counter_i]
        loc_counter_j = 0
        loc_flag_j = wrapper.Ms[loc_counter_j]
        twin_i = 0
        for i in range(wrapper.M_total):
                loc_V = np.dot(Vinf, wrapper.tgs[i])/Vmod
                U = 0
                W = 0
                twin_j = 0
                loc_counter_j = 0
                loc_flag_j = wrapper.Ms[loc_counter_j]
                for j in range(wrapper.M_total+wrapper.profile_count-1):
                    if not j == loc_flag_j:
                        r0 = [wrapper.x_points[j], wrapper.y_points[j]]
                        X_i = utis.RGLOB_LOC(np.array([wrapper.x_mid[i], wrapper.y_mid[i]]), r0, wrapper.betas[twin_j])
                        u1, u2, w1, w2 = VOR2D_L(gamma_vect[j], gamma_vect[j+1], X_i, wrapper.dL[twin_j],i,twin_j, mode = 'gen', x1 = 0)
                        u1, w1 = utis.rot_vect(utis.MAT_L2G(wrapper.betas[twin_j]), np.array([u1,w1]))
                        u2, w2 = utis.rot_vect(utis.MAT_L2G(wrapper.betas[twin_j]), np.array([u2,w2]))
                        U += u1+u2
                        W += w1+w2
                        twin_j += 1
                    else:
                        loc_counter_j += 1
                        loc_flag_j += wrapper.Ms[loc_counter_j] + 1    
                loc_V += np.dot([U,W], wrapper.tgs[i])/Vmod
                dCP[i] = 1-(loc_V**2)
                dLift[i] = 1.225*Vmod*(gamma_vect[i+twin_i]+gamma_vect[i+1+twin_i])*0.5*wrapper.dL[i]
                if i == loc_flag_i:
                    loc_counter_i += 1
                    loc_flag_i += wrapper.Ms[loc_counter_i] + 1
                    twin_i+=1
            
        return dCP, dLift