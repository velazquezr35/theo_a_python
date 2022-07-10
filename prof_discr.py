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
    def __init__(self, tipo = 'p', code = '',c = 1, gen_xcoords = None, coefs = None):
        if tipo == 'p':
            self.code = code
            self.tpoly = coefs
            self.t_coord = np.array([])
            self.denom_t = 0.2
            self.m *= c
            self.p *= c
            self.t *= c
            self.start_geom(gen_xcoords)
        elif tipo == 'f':
            self.filename = code
            self.discrete_read()
        self.c = c
        
    def start_geom(self, x_coord):
        self.y_t(x_coord)
        self.y_c(x_coord)
        self.calc_dyc(x_coord)
        self.calc_theta()
        self.calc_pos(x_coord)
        
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
        if not self.p == 0.0:
            y_part_1 = (2*self.p*x_coord-cuad_xcoord)*self.m/np.power(self.p,2)
            y_part_2 = ((1-2*self.p)+2*self.p*x_coord-cuad_xcoord)*self.m/(1-np.power(self.p,2))
            msg('test part')
            msg(y_part_1*(y_part_1<self.p))
            self.c_coord = y_part_1*(y_part_1<self.p) + y_part_2*(y_part_2>=self.p)
        else:
            self.c_coord = np.zeros(len(x_coord))
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
                
    def discrete_read(self):
        lines = open(self.filename).readlines()
        self.header = lines[0]
        self.point_count = lines[1].split()
        self.point_count = (int(self.point_count[0][:-1]),int(self.point_count[1][:-1]))
        self.XU, self.XL, self.YU, self.YL  = [],[],[],[]
        for i in range(3, self.point_count[0]+3):
            loc_line = lines[i].split()
            self.XU.append(float(loc_line[0]))
            self.YU.append(float(loc_line[1]))
        
        for i in range(self.point_count[0]+4, len(lines)):
            loc_line = lines[i].split()
            self.XL.append(float(loc_line[0]))
            self.YL.append(float(loc_line[1]))
            
    def discrete_selector(self, Ncada):
        pass
            
#Funcs
def msg(string):
    if glob_print:
        print(string)
        
#Others

#Debug / Tests