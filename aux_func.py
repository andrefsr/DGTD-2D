import numpy as np
from scipy.special import jacobi, gamma, roots_jacobi

def JacobiP(x, alpha, beta, N):
    """
    Avalia os Polinômios de Jacobi Ortonormais de grau N 
    com parâmetros alpha e beta nos pontos x.
    """
    x = np.asarray(x, dtype=float)
    
    # Gera o polinômio clássico de Jacobi usando o SciPy
    P_scipy = jacobi(N, alpha, beta)(x)
    
    # Calcula o fator de normalização gama para tornar a base ortonormal
    # (Evita overflow em ordens muito altas usando logaritmos da função gama, 
    # mas a forma direta abaixo serve para ordens moderadas usadas em Nodal DG)
    gamma_num = (2**(alpha + beta + 1)) / (2*N + alpha + beta + 1)
    gamma_den = (gamma(N + 1) * gamma(N + alpha + beta + 1))
    
    # Fator de correção
    gamma_norm = gamma_num * (gamma(N + alpha + 1) * gamma(N + beta + 1)) / gamma_den
    
    # Retorna o polinômio ortonormalizado
    return P_scipy / np.sqrt(gamma_norm)

def GradJacobiP(x, alpha, beta, N):
    """
    Avalia a derivada dos Polinômios de Jacobi Ortonormais de grau N
    com parâmetros alpha e beta nos pontos x.
    """
    x = np.asarray(x, dtype=float)
    
    # Inicializa o vetor de derivadas com zeros
    dP = np.zeros_like(x)
    
    # Se o grau for 0, o polinômio é constante, logo a derivada é 0
    if N == 0:
        return dP
    else:
        # Fator de escala derivado da relação de ortonormalidade
        scale = np.sqrt(N * (N + alpha + beta + 1.0))
        
        # A derivada é proporcional ao polinômio de grau N-1 com parâmetros +1
        dP = scale * JacobiP(x, alpha + 1.0, beta + 1.0, N - 1)
        
        return dP

def Simplex2DP(a, b, i, j):
    """Polinômios (P) bidimensionais (2D) avaliados no Triângulo (Simplex)"""
    #Evaluate 2D orthonormal polynomial on simplex at (a,b) of order (i,j).

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    # Avalia as componentes 1D
    h1 = JacobiP(a, 0, 0, i)
    h2 = JacobiP(b, 2*i + 1, 0, j)
    
    # Monta a base 2D ortonormalizada
    P = np.sqrt(2.0) * h1 * h2 * (1.0 - b)**i
    
    return P

def JacobiGL(alpha, beta, N):
    """
    Calcula os pontos (nós) de quadratura de Gauss-Lobatto de Jacobi.
    Retorna N+1 pontos no intervalo [-1, 1].
    """
    x = np.zeros(N + 1)
    
    # Se a ordem for 1, os pontos são apenas as extremidades do domínio
    if N == 1:
        x[0] = -1.0
        x[1] = 1.0
    else:
        # Define os pontos de contorno fixos
        x[0] = -1.0
        x[N] = 1.0
        
        # Calcula os N-1 pontos internos usando a função de quadratura 
        # de Gauss do SciPy com os parâmetros deslocados (+1)
        x_int, _ = roots_jacobi(N - 1, alpha + 1, beta + 1)
        
        # Preenche o miolo do vetor com os pontos internos
        x[1:N] = x_int
        
    return x

def Vandermonde1D(N,r):
    r = np.asarray(r, dtype=float)

    V1D = np.zeros((len(r),N+1))
    for j in range(N+1):
        V1D[:,j] = JacobiP(r[:],0,0,j)
    return V1D

import scipy.special as sp

def JacobiGQ(alpha, beta, N):
    '''
    Calcula os N+1 pontos (x) e pesos (w) da quadratura de Gauss-Jacobi.
    '''
    # Caso base: polinômio de ordem 0
    if N == 0:
        x = np.array([-(alpha - beta) / (alpha + beta + 2.0)])
        w = np.array([(2.0**(alpha + beta + 1.0) * sp.gamma(alpha + 1.0) * sp.gamma(beta + 1.0)) / 
                      sp.gamma(alpha + beta + 2.0)])
        return x, w

    # 1. Forma a Matriz Tridiagonal Simétrica de Jacobi (J)
    J = np.zeros((N + 1, N + 1))
    h1 = 2.0 * np.arange(N + 1) + alpha + beta

    # Diagonal principal
    # Usamos errstate para ignorar o aviso de divisão por zero que acontece no índice 0
    # quando alpha=0 e beta=0. Nós sobrescrevemos o índice 0 logo na linha seguinte.
    with np.errstate(divide='ignore', invalid='ignore'):
        diag = -0.5 * (alpha**2 - beta**2) / (h1 * (h1 + 2.0))
    diag[0] = (beta - alpha) / (alpha + beta + 2.0)
    np.fill_diagonal(J, diag)

    # Diagonais secundárias (sub e super diagonal)
    i = np.arange(1, N + 1)
    h1_i = 2.0 * i + alpha + beta
    off_diag = (2.0 / h1_i) * np.sqrt(
        (i * (i + alpha + beta) * (i + alpha) * (i + beta)) /
        ((h1_i - 1.0) * (h1_i + 1.0))
    )
    
    # Preenche as diagonais deslocadas
    idx = np.arange(N)
    J[idx, idx + 1] = off_diag
    J[idx + 1, idx] = off_diag

    # 2. Calcula Autovalores e Autovetores
    # np.linalg.eigh é especializado para matrizes simétricas (muito mais rápido e estável)
    x, V = np.linalg.eigh(J)

    # 3. Os pesos são calculados a partir da primeira linha da matriz de autovetores
    constante = (2.0**(alpha + beta + 1.0) * sp.gamma(alpha + 1.0) * sp.gamma(beta + 1.0)) / \
                sp.gamma(alpha + beta + 2.0)
    w = (V[0, :] ** 2) * constante

    return x, w