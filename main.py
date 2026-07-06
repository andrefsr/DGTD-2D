import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import aux_func as aux

def rstoab(r, s): 
    # Transfere as coordenadas de referências (r,s) para um quadrado em (a,b) onde é mais fácil criar bases ortogonais
    
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    
    a = np.zeros_like(r)
    
    mask = (s != 1.0)
    a[mask] = 2.0 * (1.0 + r[mask]) / (1.0 - s[mask]) - 1.0
    a[~mask] = -1.0
    
    b = np.copy(s)
    
    return a, b

import numpy as np

def Warpfactor(N, rout):
    # rout representa os pontos de avaliação fora/na borda
    rout = np.asarray(rout, dtype=float) 
    
    LGLr = aux.JacobiGL(0, 0, N)           # r em distribuição LGL
    req = np.linspace(-1, 1, N+1)          # r em distribuição equidistante
    
    Veq = aux.Vandermonde1D(N, req)
    
    Nr = len(rout)
    Pmat = np.zeros((N+1, Nr))

    for i in range(N+1):
        Pmat[i, :] = aux.JacobiP(rout, 0, 0, i)
        
    Lmat = np.linalg.solve(Veq.T, Pmat) #Polinômios lij de Lagrange
    
    warp = Lmat.T @ (LGLr - req)
    
    zerof = (np.abs(rout) < (1.0 - 1.0e-10))
    sf = 1.0 - (zerof * rout)**2
    
    warp = warp/sf + warp*(zerof - 1)
    
    return warp





