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

def GradJacobiP(r, alpha, beta, N):
    """
    Calcula a derivada do polinômio de Jacobi ortonormal de tipo (alpha, beta).
    """
    r_flat = np.asarray(r)
    
    # A derivada de uma constante (polinômio de ordem 0) é estritamente zero
    if N == 0:
        return np.zeros_like(r_flat)
    else:
        # Relação matemática de Hesthaven para a derivada da base ortonormal
        fator = np.sqrt(N * (N + alpha + beta + 1.0))
        dP = fator * JacobiP(r_flat, alpha + 1.0, beta + 1.0, N - 1)
        return dP
    
import numpy as np

def GradSimplex2DP(a, b, id, jd):
    """
    Retorna as derivadas espaciais (d/dr, d/ds) da base modal (id, jd)
    no simplexo 2D avaliadas nas coordenadas (a, b).
    
    Assume que as funções JacobiP e GradJacobiP estão definidas.
    """
    # Avaliações da base em 'a' e suas derivadas
    fa = JacobiP(a, 0, 0, id)
    dfa = GradJacobiP(a, 0, 0, id)
    
    # Avaliações da base ponderada em 'b' e suas derivadas
    gb = JacobiP(b, 2 * id + 1, 0, jd)
    dgb = GradJacobiP(b, 2 * id + 1, 0, jd)
    
    # --- Derivada em relação a 'r' (r-derivative) ---
    # d/dr = da/dr * d/da + db/dr * d/db = (2/(1-s)) * d/da = (2/(1-b)) * d/da
    dmodedr = dfa * gb
    if id > 0:
        dmodedr = dmodedr * ((0.5 * (1 - b)) ** (id - 1))
        
    # --- Derivada em relação a 's' (s-derivative) ---
    # d/ds = ((1+a)/2) / ((1-b)/2) * d/da + d/db
    dmodeds = dfa * (gb * (0.5 * (1 + a)))
    if id > 0:
        dmodeds = dmodeds * ((0.5 * (1 - b)) ** (id - 1))
        
    tmp = dgb * ((0.5 * (1 - b)) ** id)
    if id > 0:
        tmp = tmp - 0.5 * id * gb * ((0.5 * (1 - b)) ** (id - 1))
        
    dmodeds = dmodeds + fa * tmp
    
    # --- Normalização da Base ---
    dmodedr = (2 ** (id + 0.5)) * dmodedr
    dmodeds = (2 ** (id + 0.5)) * dmodeds
    
    return dmodedr, dmodeds

def Grad2D(u, Dr, Ds, rx, sx, ry, sy):
    """
    Calcula o campo de gradiente 2D (ux, uy) de um campo escalar u.
    
    Parâmetros:
    u  : Campo escalar avaliado nos nós.
    Dr, Ds : Matrizes de diferenciação no elemento de referência.
    rx, sx, ry, sy : Termos métricos do mapeamento geométrico.
    """
    # 1. Derivadas no triângulo de referência (Multiplicação Matricial)
    ur = Dr @ u
    us = Ds @ u
    
    # 2. Aplicação da Regra da Cadeia (Multiplicação Elemento a Elemento)
    ux = rx * ur + sx * us
    uy = ry * ur + sy * us
    
    return ux, uy

def Div2D(u, v, Dr, Ds, rx, sx, ry, sy):
    """
    Calcula a divergência 2D de um campo vetorial com componentes (u, v).
    
    Parâmetros:
    u, v : Componentes x e y do campo vetorial avaliadas nos nós.
    Dr, Ds : Matrizes de diferenciação no elemento de referência.
    rx, sx, ry, sy : Termos métricos do mapeamento geométrico.
    """
    # 1. Derivadas espaciais no triângulo de referência (operador @)
    ur = Dr @ u
    us = Ds @ u
    
    vr = Dr @ v
    vs = Ds @ v
    
    # 2. Regra da cadeia e soma para a divergência (operador *)
    # div = (du/dx) + (dv/dy)
    divu = (rx * ur + sx * us) + (ry * vr + sy * vs)
    
    return divu

def Curl2D(ux, uy, uz, Dr, Ds, rx, sx, ry, sy):
    """
    Calcula o operador rotacional (curl) 2D no plano (x,y) do campo vetorial (ux, uy, uz).
    
    Parâmetros:
    ux, uy : Componentes x e y do campo vetorial.
    uz     : Componente z do campo (pode ser None se o campo for estritamente 2D).
    Dr, Ds : Matrizes de diferenciação no elemento de referência.
    rx, sx, ry, sy : Termos métricos do mapeamento geométrico.
    """
    # 1. Derivadas espaciais de ux (operador @ para matrizes nodais)
    uxr = Dr @ ux
    uxs = Ds @ ux
    
    # 2. Derivadas espaciais de uy
    uyr = Dr @ uy
    uys = Ds @ uy
    
    # 3. Componente Z do rotacional: vz = (duy/dx) - (dux/dy)
    # Aplicando a regra da cadeia com os termos métricos (*)
    vz = (rx * uyr + sx * uys) - (ry * uxr + sy * uxs)
    
    vx = None
    vy = None
    
    # 4. Se o campo tiver uma componente transversal 'uz'
    if uz is not None:
        uzr = Dr @ uz
        uzs = Ds @ uz
        
        # Componente X do rotacional: vx = (duz/dy) - (duy/dz) -> em 2D d/dz = 0
        vx = ry * uzr + sy * uzs
        
        # Componente Y do rotacional: vy = (dux/dz) - (duz/dx) -> em 2D d/dz = 0
        vy = -(rx * uzr + sx * uzs)
        
    return vx, vy, vz