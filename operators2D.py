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

import numpy as np

def Filter2D(N, Nc, s, V):
    """
    Constrói a Matriz de Filtro Exponencial 2D.
    N  : Ordem máxima do polinômio
    Nc : Modo de corte (modos abaixo de Nc não são filtrados)
    s  : Ordem do filtro (força da atenuação, deve ser um número par, ex: 2, 4, 6...)
    """
    # alpha é definido com base no zero da máquina para que o último modo 
    # seja atenuado até o limite de precisão do computador (aprox. 36.04)
    alpha = -np.log(np.finfo(float).eps) 
    
    Np = int((N + 1) * (N + 2) / 2)
    Fdiag = np.ones(Np) # Inicializa a diagonal toda com 1 (sem filtro)
    
    sk = 0
    # Percorre todos os modos polinomiais no triângulo
    for i in range(N + 1):
        for j in range(N - i + 1):
            n = i + j # A ordem total do modo 2D é a soma i + j
            
            # Se a ordem do modo ultrapassar o corte, aplica a atenuação exponencial
            if n >= Nc:
                Fdiag[sk] = np.exp(-alpha * ((n - Nc) / (N - Nc))**s)
                
            sk += 1
            
    # Monta a matriz diagonal Lambda
    Lambda = np.diag(Fdiag)
    
    # Constrói a Matriz de Filtro F = V * Lambda * V^-1
    F = V @ Lambda @ np.linalg.inv(V)
    
    return F

import numpy as np

def Lift2D(N, r, s, V):
    '''Compute surface to volume lift term for DG formulation'''
    # Quando a simulação estiver rodando no loop de tempo,
    # você vai calcular a diferença da onda entre um triângulo e seu vizinho (o fluxo na borda) e multiplicar por essa matriz LIFT.
    # Ela automaticamente espalhará a correção do fluxo para os nós internos do triângulo
    
    #Constrói a matriz de índices Fmask para o triângulo de referência.
    #Retorna uma matriz onde cada coluna contém os índices dos nós 
    #pertencentes a uma das faces.

    tol=1e-10

    # 1. Encontra os índices (IDs) dos pontos que satisfazem a geometria da face
    # O [0] no final serve para extrair o array de dentro da tupla que o np.where retorna
    fmask1 = np.where(np.abs(s + 1.0) < tol)[0]  # Face 1: s = -1
    fmask2 = np.where(np.abs(r + s) < tol)[0]    # Face 2: r + s = 0
    fmask3 = np.where(np.abs(r + 1.0) < tol)[0]  # Face 3: r = -1
    
    # 2. Agrupa as três listas como colunas em uma matriz 2D
    # O resultado será uma matriz de tamanho (Nfp, 3)
    Fmask = np.column_stack((fmask1, fmask2, fmask3))

    Np = ((N + 1) * (N + 2)) // 2  # Divisão inteira
    Nfp = N + 1                    # Número de nós na face (borda)
    Nfaces = 3                     # Triângulo tem 3 faces
    
    Emat = np.zeros((Np, Nfaces * Nfp))
    
    # --- FACE 1 (Aresta Inferior) ---
    faceR = r[Fmask[:, 0]]
    V1D = aux.Vandermonde1D(N, faceR)
    massEdge1 = np.linalg.inv(V1D @ V1D.T)
    # Insere os blocos na matriz usando a indexação correta do Python
    Emat[Fmask[:, 0], 0:Nfp] = massEdge1
    
    # --- FACE 2 (Aresta Direita) ---
    faceR = r[Fmask[:, 1]]
    V1D = aux.Vandermonde1D(N, faceR)
    massEdge2 = np.linalg.inv(V1D @ V1D.T)
    Emat[Fmask[:, 1], Nfp:2*Nfp] = massEdge2
    
    # --- FACE 3 (Aresta Esquerda) ---
    faceS = s[Fmask[:, 2]] # Aqui usamos s, pois r é constante na face esquerda
    V1D = aux.Vandermonde1D(N, faceS)
    massEdge3 = np.linalg.inv(V1D @ V1D.T)
    Emat[Fmask[:, 2], 2*Nfp:3*Nfp] = massEdge3
    
    # LIFT = Matriz de Massa Inversa * Emat
    LIFT = V @ (V.T @ Emat)
    
    return LIFT






    
