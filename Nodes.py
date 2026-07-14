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

def Nodes2D(N):
    # obtém as coordenadas (x,y) dos nós no triângulo equilátero de referencia.

    alpopt = [0.0 , 0.0 , 1.4152, 0.1001, 0.2751, 0.98, 1.099,
              1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]

    if N < 15:
        alpha = alpopt[N]
    else:
        alpha = 5/3
    
    Np = int(((N+1)*(N+2))/2.0)

    L1 = np.zeros(Np)
    L2 = np.zeros_like(L1)
    L3 = np.zeros_like(L1)

    sk = 0
    for n in range(N+1):
        for m in range(N+1-n):
            L1[sk] = n/N
            L3[sk] = m/N
            sk += 1
    L2 = 1.0 - L1 - L3
    x = -L2 + L3
    y = (-L2 - L3 + 2*L1)/np.sqrt(3.0)

    blend1 = 4*L2*L3
    blend2 = 4*L1*L3
    blend3 = 4*L1*L2

    warpf1 = Warpfactor(N,L3-L2)
    warpf2 = Warpfactor(N,L1-L3)
    warpf3 = Warpfactor(N,L2-L1)

    warp1 = blend1*warpf1*(1+(alpha*L1)**2)
    warp2 = blend2*warpf2*(1+(alpha*L2)**2)
    warp3 = blend3*warpf3*(1+(alpha*L3)**2)

    x = x + 1*warp1 + np.cos(2*(np.pi/3))*warp2 + np.cos(4*(np.pi/3))*warp3
    y = y + 0*warp1 + np.sin(2*(np.pi/3))*warp2 + np.sin(4*(np.pi/3))*warp3

    return x, y

def xytors(x,y):
    # Transforma os nós definidos no triangulo equilátero no triangulo de referencia em r,s

    L1 = (np.sqrt(3.0)*y + 1.0)/3.0 # Lambda 1 das coordenadas baricêntricas
    L2 = (-3.0*x - np.sqrt(3.0)*y + 2.0)/6.0
    L3 = ( 3.0*x - np.sqrt(3.0)*y + 2.0)/6.0

    r = -L2 + L3 - L1
    s = -L2 - L3 + L1

    return r, s