"""
03/07/2022

Discretización de perfiles y tratamiento geométrico

@author: Rodrigo Ramón Velazquez
"""

#Imports
import numpy as np

#Global controls
glob_print = True

#Classes

class NACA4Code:
    def __set__(self, obj, code):
        if not len(code)==4:
            raise ValueError(f'Código NACA debe ser serie 4: {code}')
        else:
            try:
                obj.m = float(code[0])/100
                obj.p = float(code[1])/10
                obj.t = float(code[2:])/100
            except ValueError:
                print('Código NACA debe ser inteable')
                raise
            else:
                obj.name = 'NACA ' + code
                msg('CODE OK')
                msg(f'PROFILE NACA {code}')
                msg(f'm = {obj.m}')
                msg(f'p = {obj.p}')
                msg(f't = {obj.t}')

class NACA4tPoly:
    '''
    TEMP
    Por defecto, descriptor de un poly 05 1 2 3 4 
    '''
    
    def __set__(self, obj, coefs = None):
        '''
        TEMP
        coefs may be (), [] o np.array
        '''
        if coefs is None:
            msg(f'{obj.name} - Seteando coeficientes por defecto')
            obj.tpow_coefs = np.array([0.5, 1, 2, 3, 4])
            obj.tmul_coefs = np.array([0.2969, -0.126, -0.3516, 0.2843, -0.1015])
        else:
            msg(f'{obj.name} - Seteando coeficientes indicados, n = {len(coefs)}')
            obj.tpow_coefs = np.array(coefs[0])
            obj.tmul_coefs = np.array(coefs[1])
    
class NACA4Prof:
    code = NACA4Code()
    tpoly = NACA4tPoly()
    def __init__(self, code = '',c = 1, coefs = None):
        self.code = code
        self.tpoly = coefs
        self.t_coord = np.array([])
        self.denom_t = 0.2
        self.m *= c
        self.p *= c
        self.t *= c
        self.c = c
        
    def y_t(self, x_coord):
        '''
        TEMP
        producto interno de evaluacion de polinomio de espesor
        x_coord es un float, int o np array 1XN
        '''
        if isinstance(x_coord, int) or isinstance(x_coord, float):
            self.t_coord = np.sum(np.power(x_coord, self.tpow_coefs)*self.tmul_coefs)
        else:
            if not isinstance(x_coord, np.ndarray):
                x_coord = np.array(x_coord)
            self.t_coord = np.sum(np.power(x_coord.reshape((len(x_coord), 1)), self.tpow_coefs)*self.tmul_coefs, axis = 1)       
        self.t_coord = self.t_coord*self.t/self.denom_t
        msg(self.t_coord)
        
    def y_c(self, x_coord):
        '''
        TEMP
        producto interno de evaluacion de y camber
        '''
        if not isinstance(x_coord, np.ndarray):
            x_coord = np.array(x_coord)
        #parte 1
        cuad_xcoord = np.power(x_coord,2)
        y_part_1 = (2*self.p*x_coord-cuad_xcoord)*self.m/np.power(self.p,2)
        y_part_2 = ((1-2*self.p)+2*self.p*x_coord-cuad_xcoord)*self.m/(1-np.power(self.p,2))
        msg('test part')
        msg(y_part_1*(y_part_1<self.p))
        self.c_coord = y_part_1*(y_part_1<self.p) + y_part_2*(y_part_2>=self.p)
        msg(self.c_coord)
        
    def calc_dyc(self, x_coord):
        self.dyc = np.gradient(self.c_coord)
        self.dx = np.gradient(x_coord)
        msg(self.dyc)
        
    def calc_theta(self):
        self.theta = np.arctan(self.dyc/self.dx)
    
    def calc_pos(self, x_coord):
        
        loc_presin = self.t_coord*np.sin(self.theta)
        loc_precos = self.t_coord*np.cos(self.theta)
        
        self.XU = x_coord - loc_presin
        self.XL = x_coord + loc_presin
        self.YL = self.c_coord - loc_precos
        self.YU = self.c_coord + loc_precos
#Funcs
def msg(string):
    if glob_print:
        print(string)
        
def prof_discretizer():
    pass
#Others

#Debug / Tests

# ramon = NACA4Prof('4412')

# x_coord = np.linspace(0,1,4)
# ramon.y_t(x_coord)
# ramon.y_c(x_coord)
# ramon.calc_dyc(x_coord)
# ramon.calc_theta()
# ramon.calc_pos(x_coord)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# ax.plot(ramon.XU, ramon.YU, 'ro')
# ax.plot(ramon.XL, ramon.YL, 'bo')
# ax.set_aspect('equal')