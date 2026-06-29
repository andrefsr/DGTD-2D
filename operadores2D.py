import numpy as np
from scipy.special import gamma

def JacobiP(x, alpha, beta, N):
    """
    Avalia o Polinômio de Jacobi ortonormal de tipo (alpha,beta) > -1
    nos pontos x para a ordem N.
    """
    #xp = np.atleast_1d(x)
    xp = np.asarray(x).flatten() # Força a virar (66,) removendo dimensões extras
    PL = np.zeros((N + 1, len(xp)))

    # Valores iniciais P_0(x) e P_1(x)
    gamma0 = (2**(alpha + beta + 1) / (alpha + beta + 1)) * \
             (gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1))
    
    PL[0, :] = 1.0 / np.sqrt(gamma0)
    if N == 0:
        return PL[0, :]

    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[1, :] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)
    
    if N == 1:
        return PL[1, :]

    # Relação de recorrência para as próximas ordens
    aold = 2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))

    for i in range(1, N):
        h1 = 2 * i + alpha + beta
        anew = 2 / (h1 + 2) * np.sqrt((i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta) / ((h1 + 1) * (h1 + 3)))
        bnew = - (alpha**2 - beta**2) / (h1 * (h1 + 2))

        # PL[i+1] equivale ao PL(i+2) do MATLAB devido ao índice 0
        PL[i + 1, :] = 1 / anew * (-aold * PL[i - 1, :] + (xp - bnew) * PL[i, :])
        aold = anew

    return PL[N, :]

def JacobiGQ(alpha, beta, N):
    """
    Calcula os pontos de quadratura de Gauss e seus respectivos pesos.
    """
    if N == 0:
        x = np.array([-(alpha - beta) / (alpha + beta + 2)])
        w = np.array([2.0])
        return x, w

    # Forma a matriz tridiagonal simétrica (Matriz de Jacobi)
    J = np.zeros((N + 1, N + 1))
    h1 = 2 * np.arange(N + 1) + alpha + beta

    # Diagonal principal
    np.fill_diagonal(J, -0.5 * (alpha**2 - beta**2) / ((h1 + 2) * h1))
    
    # Subdiagonal
    i = np.arange(1, N + 1)
    subdiag = (2 / (h1[0:N] + 2)) * np.sqrt((i * (i + alpha + beta) * (i + alpha) * (i + beta)) / ((h1[0:N] + 1) * (h1[0:N] + 3)))
    
    # Preenche as subdiagonais superior e inferior (tornando-a simétrica)
    np.fill_diagonal(J[1:], subdiag)
    np.fill_diagonal(J[:, 1:], subdiag)

    if alpha + beta < 10 * np.finfo(float).eps:
        J[0, 0] = 0.0

    # Os pontos (raízes) são os autovalores da Matriz de Jacobi
    D, V = np.linalg.eig(J)
    
    # É importante ordenar os autovalores para uso em métodos espectrais
    idx = D.argsort()
    x = D[idx]
    V = V[:, idx]

    # Pesos da quadratura
    w = (2**(alpha + beta + 1) / (alpha + beta + 1)) * \
        (gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1)) * (V[0, :]**2)

    return x, w
def JacobiGL(alpha, beta, N):
    """
    Calcula os pontos de quadratura de Gauss-Lobatto de ordem N.
    """
    if N == 1:
        return np.array([-1.0, 1.0])

    # Calcula os pontos internos usando a função de Gauss
    xint, _ = JacobiGQ(alpha + 1, beta + 1, N - 2)

    # Adiciona os extremos -1 e 1 concatenando os arrays
    x = np.concatenate(([-1.0], xint, [1.0]))
    
    return x    
def Vandermonde1D(N, r):
    """
    Inicializa a matriz de Vandermonde 1D generalizada, V_{ij} = P_j(r_i),
    onde P_j é o polinômio de Jacobi ortonormal (Legendre).
    """
    r_flat = np.atleast_1d(r)
    V1D = np.zeros((len(r_flat), N + 1))
    
    # Preenche cada coluna com o polinômio de ordem correspondente
    for j in range(N + 1):
        V1D[:, j] = JacobiP(r_flat, 0, 0, j)
        
    return V1D

import numpy as np

def Simplex2DP(a, b, i, j):
    """
    Avalia o polinômio ortonormal 2D (base PKD) no triângulo de referência 
    simplexo nas coordenadas (a,b) para a ordem (i,j).
    
    Assume que a função JacobiP(x, alpha, beta, N) está definida no escopo.
    """
    # Avalia o polinômio de Jacobi em 'a' (Polinômio de Legendre)
    h1 = JacobiP(a, 0, 0, i)
    
    # Avalia o polinômio de Jacobi ponderado em 'b'
    h2 = JacobiP(b, 2 * i + 1, 0, j)
    
    # Combina para formar a base de Proriol-Koornwinder-Dubiner (PKD)
    P = np.sqrt(2.0) * h1 * h2 * ((1 - b) ** i)
    
    return P