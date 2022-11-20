import geom_discr
import pot_flow
import plotter
import math_tools as utis


# Testing
if True:
    
    import main_aero
    import numpy as np
    # gg = np.linspace(np.radians(0), np.radians(180), 90)
    # x_coord = 0.5*(1-np.cos(gg))
    # x_coord = np.linspace(0,1,91)
    # ramon = main_aero.aero_prof(0,'p', '0012', 1, x_coord)
    # ramon.spatial_discretization()
    
    

    # prof = geom_discr.Prof_gen(from_coords = True, x_points = ramon.panel_coords[:-1,0], y_points = ramon.panel_coords[:-1,1])

    # simple_x = [1,0,-0.5,0]
    # simple_y = [0, -0.5, 0, 0.5]
    tt = np.linspace(np.radians(0), np.radians(360),99)
    simple_x = np.cos(tt)
    simple_y = np.sin(tt)
    dg = np.radians(0)
    mod = 1
    Vinf = [mod*np.cos(dg), mod*np.sin(dg)]
    # simple_x = [1,0,0,1]
    # simple_y = [0,0,1,1]
    
    prof = geom_discr.Prof_gen(from_coords = True, x_points=simple_x, y_points = simple_y)
    prof.update()
    if True:
        inf_mat, RHS = pot_flow.GEN_INF_MATRX(prof, Vinf)
        gamma_vect = pot_flow.GAMMA_PROD(inf_mat, RHS)
        CPs = pot_flow.CP(prof, gamma_vect, Vinf)
        dL, L = pot_flow.CL(prof, gamma_vect, Vinf, 1.225)
        
    if True:
        fig, ax = plotter.plot_tester(prof)
        ax.plot(prof.x_points[0], prof.y_points[0], 'ro')
        fig2, ax2 = plotter.plot_CPs(prof, CPs)
        ax2.plot(prof.x_mid[-1], CPs[-1],'ro')
        ax2.plot(prof.x_mid[0], CPs[0],'bo')
        fig3, ax3 = plotter.plot_dLift(prof, dL)
        ax3.plot(prof.x_mid[-1], dL[-1], 'ro')
        fig4, ax4 = plotter.plot_gammas(prof, gamma_vect[:,0])
        #plotter.test_PL_point(ax, prof, [prof.dL[3]*0.5,0], 3)
        
    if True:
        print('Printing normal comp over CPoints:')
        for i in range(prof.M):
            loc_u, loc_w = pot_flow.EVAL_FIELD(prof.x_mid[i], prof.y_mid[i], prof, gamma_vect, Vinf)
            print(np.dot([loc_u, loc_w], prof.norms[i]))
            
    if False:
        size = 100
        X, Y = utis.QUAD_GRID([-5,5,-5,5], size)
        U, W = pot_flow.FLOW_FIELD(X,Y, prof, gamma_vect, Vinf, size, alone = False)
        for i in range(size):
            for j in range(size):
                if np.sqrt(X[i,j]**2 + Y[i,j]**2) < 1:
                    U[i,j] = 0
                    W[i,j] = 0
        # plotter.plot_Vmap(ax, X, Y, U, W)
        plotter.plot_field(ax,X,Y,U,W, density = 5)
        


# 
# )

# 

# u = np.zeros((size,size))
# w = np.zeros((size,size))

# for i in range(size):
#     for j in range(size):
#         u[i,j],w[i,j] = Singulars.VLD_2D(X[i,j], Y[i,j], gamma_1, gamma_2, x2)
        
# fig,ax = plt.subplots(dpi = 100)
 
# # Plotting stream plot

# Z = np.sqrt(u**2+w**2)
# ax.imshow(Z, interpolation='bilinear', extent = [x[0], x[-1], y[0], y[-1]])
# 
# ax.plot([0,x2],[0,0], linewidth = 5)