import numpy as np

def LST_ARR(tent_lst):
    if isinstance(tent_lst, list):
        return np.array(tent_lst)
    else:
        return tent_lst
    
def checkabs(*vals):
    ret = []
    for loc in vals:
        if abs(loc) < 1e-10:
            loc = 0
        ret.append(loc)
    return(ret)

def checkarr(gamma_1, gamma_2):
    if isinstance(gamma_1, np.ndarray):
        gamma_1 = gamma_1[0]
    if isinstance(gamma_2,np.ndarray):
        gamma_2 = gamma_2[0]
    return gamma_1, gamma_2
    
def QUAD_GRID(coords, size):
    x = np.linspace(*coords[0:2], size)
    y = np.linspace(*coords[1:-1], size)
    return np.meshgrid(x,y)

def MAT_G2L(beta):
    return np.array([[np.cos(beta), np.sin(beta)],[-np.sin(beta), np.cos(beta)]])

def MAT_L2G(beta):
    return np.array([[np.cos(beta), -np.sin(beta)],[np.sin(beta), np.cos(beta)]])

def RGLOB_LOC(P, r0, beta):
    r_g = P - r0
    r_l = np.matmul(MAT_G2L(beta), r_g)
    return r_l

def RLOC_GLOB(P_l, r0, beta):
    r_g = r0 + np.matmul(MAT_L2G(beta), P_l)
    return r_g
       
def appnd_last(arr):
    return np.append(arr, arr[0])

def rot_vect(matrx, vec):
    return np.matmul(matrx, vec)

def gen_norm(theta, v0, scale=1):
    probe_t = rot_vect(MAT_L2G(theta), np.array([1,0])*scale)
    probe_n = rot_vect(MAT_L2G(theta), np.array([0,1])*scale)
    print(probe_n)
    print(v0)
    print(np.dot(probe_t, probe_n))
    p_1x = probe_t + v0
    p_1y = probe_n + v0
    print(p_1x, p_1y)
    print(np.dot(p_1x, p_1y))
    print('s')
    return(v0, p_1x, p_1y)