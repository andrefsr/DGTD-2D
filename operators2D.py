import numpy as np
import aux_func as aux
import Nodes

def Vandermonde2D(N,r,s):
    # Inivializa a matriz de Vandermonde onde
    # V_{ij} = phi_j(r_i,s_i)

    V2D = np.zeros(((len(r)),int(((N+1)*(N+2))/2)))

    a, b = Nodes.rstoab(r,s)

    sk = 0
    for i in range(N+1):
        for j in range(N-i+1):
            V2D[:,sk] =  aux.Simplex2DP(a,b,i,j)
            sk += 1
    
    return V2D

def GradSimplex2DP(a, b, i_d, j_d):
    
    fa = aux.JacobiP(a, 0, 0, i_d)
    dfa = aux.GradJacobiP(a, 0, 0, i_d)
    gb = aux.JacobiP(b, 2*i_d + 1, 0, j_d)
    dgb = aux.GradJacobiP(b, 2*i_d + 1, 0, j_d)

    # Derivada em relação a r
    #d/dr = da/dr d/da + db/dr d/db = (2/(1-s)) d/da = (2/(1-b)) d/da dmodedr = dfa.*gb;
    dmodedr = dfa * gb
    if i_d > 0:
        dmodedr = dmodedr * ((0.5 * (1 - b))**(i_d - 1))

    # Derivada em relação a s
    # d/ds = ((1+a)/2)/((1-b)/2) d/da + d/db
    dmodeds = dfa * (gb * (0.5 * (1 + a)))
    if i_d > 0:
        dmodeds = dmodeds * ((0.5 * (1 - b))**(i_d - 1))

    tmp = dgb * ((0.5 * (1 - b))**i_d)
    if i_d > 0:
        tmp = tmp - 0.5 * i_d * gb * ((0.5 * (1 - b))**(i_d - 1))

    dmodeds = dmodeds + fa * tmp

    # Normalização
    dmodedr = (2**(i_d + 0.5)) * dmodedr
    dmodeds = (2**(i_d + 0.5)) * dmodeds

    return dmodedr, dmodeds

def GradVandermonde2D(N,r,s):
    '''Inicializa o gradiente da base modal em r,s de ordem N'''

    Np = int(((N+1)*(N+2))/2)

    V2Dr = np.zeros((len(r),Np))
    V2Ds = np.zeros_like(V2Dr)

    a, b = Nodes.rstoab(r,s)

    sk = 0
    for i in range(N+1):
        for j in range(N+1-i):
            V2Dr[:,sk], V2Ds[:,sk] = GradSimplex2DP(a,b,i,j)
            sk += 1

    return V2Dr, V2Ds 

def Dmatrices2D(N,r,s,V):
    """Matriz de diferenciação"""

    Vr, Vs = GradVandermonde2D(N,r,s)
    invV = np.linalg.inv(V)
    Dr = Vr @ invV
    Ds = Vs @ invV

    return Dr, Ds


