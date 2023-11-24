"""
#Theo_a_python
A simple tool for analysis of 2D airfoils using potential flow theory
Methodology: Linear vorticity distribution
Version: Python prepro, solver & postpro

@author: Rodrigo R. Velazquez

AUG 2023

#Module
Testing

- Case testing and validation
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt
import geom_discr
import pot_flow
import plotter


#Auxiliar funcs
def triangular(x, p=0.25, t=0.1, c = 1):
    '''
    A simple shape generator for normal and points visualization and coefficient check
    
    Inputs:
        x - Array like x-coord distribution
    Optional:
        p - TBD
        t - TBD
        c - TBD
    Returns:
        y - Array like external y-coord surface distribution
    '''
    y_e = []
    change = True
    for i in range(len(x)):
        if x[i]<=p*c:
            y_e.append(x[i]*t/p)
        else:
            if change:
                x_change = x[i-1]
                alt_change = y_e[-1]
                change = False
            pend = -alt_change/(x[-1]-x_change)
            y_e.append(alt_change + pend*(x[i]-x_change))
    y_e = np.array(y_e)
    y_e2 = -y_e[1:-1]
    x2 = x[1:-1]
    x = np.flip(x)
    y_e = np.flip(y_e)
    return -np.append(y_e,np.flip(y_e2)), np.append(x,x2)


def yp_NACA0012(x):
    '''
    NACA 0012 external surface generator y(x)
    
    Inputs: x - Array-like x coord distribution
    
    Returns: y_p - Array-like y extrados and intrados coord distribution
    '''
    y_p = 0.594689181*(0.298222773*np.sqrt(x) - 0.127125232*x - 0.357907906*(x**2) + 0.291984971*(x**3) - 0.105174606*(x**4))
    return(y_p/1.008930411365)

#Main func

def NACA0012_validation():
    
    '''
    NACA 0012 validation case against XFOIL output data
    Computes a NACA 0012 CPvsX for alpha 8 deg and the CLvsalpha curve
    
    External data file should be indicated
    
    Inputs:
        
    Optional:
        
    Outpus:
        
    '''
        
    #NACA 0012 alpha 8 deg validation case
    
    ##Set local values
    number_of_points = 50
    external_flow_Vinf = 1
    air_density = 1.225
    #CPvsX data
    main_alpha = 8
    
    ##Read external data
    validation_alphas, validation_CLs = geom_discr.read_txt_xfoil(r'D:\test\0012RE500E3.txt', header_init = 12)
    validation_x, validation_cp = geom_discr.read_txt_xfoil(r'D:\test\0012a8.txtt', header_init=1)
    
    ##NACA0012 profile generation
    ###Cosine geom. discretization
    dist_x = np.linspace(np.radians(180), np.radians(0), number_of_points)
    dist_x = 0.5*(1-np.cos(dist_x))
    dist_y = yp_NACA0012(dist_x)
    ###Profile obj init
    x_profile = np.append(dist_x, np.flip(dist_x[1:-1]))
    y_profile = np.append(-dist_y, np.flip(dist_y[1:-1]))
    NACA0012_prof = geom_discr.Prof_gen(from_coords = True, x_points = x_profile, y_points = y_profile)
    NACA0012_prof.update()

    #CPvsX calculation
    main_alpha_rads = np.radians(main_alpha)
    Vinf = [external_flow_Vinf*np.cos(main_alpha_rads), external_flow_Vinf*np.sin(main_alpha_rads)]

    inf_mat, RHS, tg_mat = pot_flow.GEN_INF_MATRX(NACA0012_prof, Vinf)
    gamma_vect = pot_flow.GAMMA_PROD(inf_mat, RHS)
    CPs, Lifts = pot_flow.CPCL(NACA0012_prof, gamma_vect,Vinf)
    
    
    ###Postpro
    fig_CP, ax_CP = plotter._testing_CPvs(NACA0012_prof, CPs, [validation_x, validation_cp])
    
    #CLvsalpha calculation
    CL_lst = []
    for loc_alpha in validation_alphas:
        loc_alpha = np.radians(loc_alpha)
        loc_Vinf = [external_flow_Vinf*np.cos(loc_alpha), external_flow_Vinf*np.sin(loc_alpha)]
        loc_inf_mat, loc_RHS, loc_tg_mat = pot_flow.GEN_INF_MATRX(NACA0012_prof, loc_Vinf)
        loc_gamma_vect = pot_flow.GAMMA_PROD(loc_inf_mat, loc_RHS)
        loc_CPs, loc_Lifts = pot_flow.CPCL(NACA0012_prof, loc_gamma_vect,loc_Vinf)
        CL_lst.append(sum(loc_Lifts)*2/(air_density*1*external_flow_Vinf**2)) 
    
    ###Postpro
    fig_CLs, ax_CLs = plotter._testing_CLvs([validation_alphas,validation_CLs], CL_lst)
    plotter.plot_tester(NACA0012_prof)
#Standalone run

# if __name__ == '__main__':
#     import time
#     start = time.time()
#     NACA0012_validation()
#     end = time.time()
#     print('Elapsed [s]: ')
#     print(end - start)

if __name__ == '__main__':
    
    #NACA 0012 alpha 8 deg validation case
    
    ##Set local values
    number_of_points = 50
    external_flow_Vinf = 1
    air_density = 1.225
    #CPvsX data
    main_alpha = 8
    
    ##NACA0012 profile generation
    ###Cosine geom. discretization
    dist_x = np.linspace(np.radians(180), np.radians(0), number_of_points)
    dist_x = 0.5*(1-np.cos(dist_x))
    dist_y = yp_NACA0012(dist_x)
    ###Profile obj init
    x_profile = np.append(dist_x, np.flip(dist_x[1:-1]))
    y_profile = np.append(-dist_y, np.flip(dist_y[1:-1]))
    NACA0012_prof = geom_discr.Prof_gen(from_coords = True, x_points = x_profile, y_points = y_profile)
    NACA0012_prof.update()
    
    FLAP_prof = geom_discr.Prof_gen(from_coords = True, x_points = x_profile, y_points = y_profile, traslation = [1.01, -0.1])
    FLAP_prof.update()
    
    WRAPPER = geom_discr.Assy_wrapper()
    WRAPPER.add_profile(NACA0012_prof)
    WRAPPER.add_profile(FLAP_prof)
    WRAPPER.assy_mtrxs()
    
    plotter.plot_tester([NACA0012_prof, FLAP_prof])
    
    main_alpha_rads = np.radians(main_alpha)
    Vinf = [external_flow_Vinf*np.cos(main_alpha_rads), external_flow_Vinf*np.sin(main_alpha_rads)]
    
    inf_mat, RHS, tg_mat = pot_flow.GEN_INF_MATRX(WRAPPER, Vinf)
    gamma_vect = pot_flow.GAMMA_PROD(inf_mat, RHS)
    CPs, Lifts = pot_flow.CPCL(WRAPPER, gamma_vect,Vinf)
    CL = sum(Lifts)*2/(air_density*1*external_flow_Vinf**2)
    print(CL)
    

    
    #Do not use, so slow...
    # size = int(2E3)
    # X = np.linspace(-0.1,1.3, size)
    # Y = np.linspace(-0.4,0.4, size)
    # U,W = pot_flow.FLOW_FIELD(X, Y, WRAPPER, gamma_vect, Vinf,size, False)
    # X,Y = np.meshgrid(X,Y)
    # fig2, ax2 = plt.subplots()
    # plotter.plot_Vmap(ax2, X, Y, U, W)