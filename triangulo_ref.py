import numpy as np  
import matplotlib.pyplot as plt
from scipy.special import gamma
import operadores2D as op2D

def rstoab(r, s):
    """
    Transforma do triângulo de referência (r,s) para o domínio colapsado (a,b).
    Inclui tratamento rigoroso para a singularidade em s = 1.
    """
    r_flat = np.asarray(r).flatten()
    s_flat = np.asarray(s).flatten()
    
    a = np.zeros_like(r_flat)
    
    # Identifica o vértice superior para evitar divisão por zero
    is_top = np.isclose(s_flat, 1.0, atol=1e-11)
    
    # Mapeia os pontos normais
    a[~is_top] = 2.0 * (1.0 + r_flat[~is_top]) / (1.0 - s_flat[~is_top]) - 1.0
    
    # Isola a singularidade (força a = -1)
    a[is_top] = -1.0
    
    b = s_flat.copy()
    
    return a, b

def Warpfactor(N, rout):
    LGLr = op2D.JacobiGL(0,0,N)
    req = np.linspace(-1,2,N+1) ## r equidistante depois é mapeado em gll

    Veq = op2D.Vandermonde1D(N,req)

    Nr = len(rout)
    Pmat = np.zeros((N+1,Nr))
    for i in range(0,N+1):
        Pmat[i,:] = op2D.JacobiP(rout,0,0,i)

    Lmat = np.linalg.solve(Veq.T, Pmat)

    warp = Lmat.T @ (LGLr - req)

    zerof = np.abs(rout) < (1.0 - 1e-10)
    sf = 1.0 - (zerof * rout)**2
    warp = (warp / sf) + (warp * (zerof - 1))

    return warp

def Nodes2D(N):
    """
    Calcula os nós (x,y) num triângulo equilátero para um polinómio de ordem N.
    Assume que a função Warpfactor(N, rout) já está definida no mesmo âmbito.
    """
    # Parâmetro otimizado alpha, dependendo da ordem N
    alpopt = np.array([
        0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 
        1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258
    ])
    
    if N < 16:
        # N-1 porque o MATLAB acede ao 1º elemento com (1) e o Python com [0]
        alpha = alpopt[N - 1] 
    else:
        alpha = 5.0 / 3.0
        
    # Número total de nós
    Np = int((N + 1) * (N + 2) / 2)
    
    # Criar nós equidistribuídos no triângulo equilátero
    L1 = np.zeros(Np)
    L2 = np.zeros(Np)
    L3 = np.zeros(Np)
    
    sk = 0 # O índice em Python começa em 0
    
    # Loops ajustados para os índices do Python (range(a, b) para antes de b)
    for n in range(1, N + 2):
        for m in range(1, N + 3 - n):
            L1[sk] = (n - 1) / N
            L3[sk] = (m - 1) / N
            sk += 1
            
    L2 = 1.0 - L1 - L3
    
    # Transformação para coordenadas Cartesianas (x, y) base
    x = -L2 + L3
    y = (-L2 - L3 + 2 * L1) / np.sqrt(3.0)
    
    # Calcular a função de mistura (blending) em cada nó para cada aresta
    blend1 = 4 * L2 * L3
    blend2 = 4 * L1 * L3
    blend3 = 4 * L1 * L2
    
    # Quantidade de warp (distorção) para cada nó, por cada aresta
    warpf1 = Warpfactor(N, L3 - L2)
    warpf2 = Warpfactor(N, L1 - L3)
    warpf3 = Warpfactor(N, L2 - L1)
    
    # Combinar a mistura (blend) e a distorção (warp)
    warp1 = blend1 * warpf1 * (1 + (alpha * L1)**2)
    warp2 = blend2 * warpf2 * (1 + (alpha * L2)**2)
    warp3 = blend3 * warpf3 * (1 + (alpha * L3)**2)
    
    # Acumular as deformações associadas a cada aresta
    x = x + 1.0 * warp1 + np.cos(2 * np.pi / 3) * warp2 + np.cos(4 * np.pi / 3) * warp3
    y = y + 0.0 * warp1 + np.sin(2 * np.pi / 3) * warp2 + np.sin(4 * np.pi / 3) * warp3
    
    return x, y

def xytors(x,y):
    L1 = (np.sqrt(3.0)*y+1.0)/3.0
    L2 = (-3.0*x - np.sqrt(3.0)*y + 2.0)/6.0
    L3 = (3.0*x - np.sqrt(3.0)*y + 2.0)/6.0

    r = -L2 + L3 - L1
    s = -L2 - L3 + L1
    return r, s

def Vandermonde2D(N, r, s):

    V2D = np.zeros((len(r),int((N+1)*(N+2)/2)))

    a, b = rstoab(r,s) 

    sk = 0
    for i in range(N+1):
        for j in range(N - i + 1):
            V2D[:,sk] = op2D.Simplex2DP(a,b,i,j)
            sk += 1
    return V2D