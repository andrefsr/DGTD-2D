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

