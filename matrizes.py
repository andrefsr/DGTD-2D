import operadores2D as op2D
import matplotlib.pyplot as plt
import triangulo_ref as tref
import numpy as np
import scipy.sparse as sp

def Vandermonde2D(N, r, s):

    V2D = np.zeros((len(r),int((N+1)*(N+2)/2)))

    a, b = tref.rstoab(r,s) 

    sk = 0
    for i in range(N+1):
        for j in range(N - i + 1):
            V2D[:,sk] = op2D.Simplex2DP(a,b,i,j)
            sk += 1
    return V2D

import numpy as np

def GradVandermonde2D(N, r, s):
    """
    Inicializa o gradiente da base modal (i,j) avaliado nos nós (r,s) 
    para um polinômio de ordem N.
    Retorna as matrizes de Vandermonde das derivadas V2Dr e V2Ds.
    """
    # Número total de nós/polinômios
    Np = int((N + 1) * (N + 2) / 2)
    Nr = len(np.atleast_1d(r))
    
    # Inicializa as matrizes com zeros (usando a tupla de dimensões!)
    V2Dr = np.zeros((Nr, Np))
    V2Ds = np.zeros((Nr, Np))
    
    # Encontra as coordenadas do domínio colapsado
    a, b = tref.rstoab(r, s)
    
    # Preenche as colunas das matrizes
    sk = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            # A função GradSimplex2DP retorna os gradientes em 'r' e 's'
            V2Dr[:, sk], V2Ds[:, sk] = op2D.GradSimplex2DP(a, b, i, j)
            sk += 1
            
    return V2Dr, V2Ds
    
def Dmatrices2D(N,r,s,V):
    Vr, Vs = GradVandermonde2D(N,r,s)
    Dr = Vr/V 
    Ds = Vs/V
    return Dr, Ds

def Filter2D(Norder, Nc, sp, V, invV):
    """
    Inicializa a matriz de filtro 2D exponencial de ordem 'sp' e corte 'Nc'.
    
    Parâmetros:
    Norder : Ordem máxima do polinômio (N).
    Nc     : Modo de corte (cutoff) a partir do qual o filtro começa a atuar.
    sp     : Ordem do filtro (define o quão "abrupto" é o decaimento).
    V      : Matriz de Vandermonde 2D.
    invV   : Inversa da Matriz de Vandermonde 2D.
    """
    # Calcula o número total de nós/modos
    Np = int((Norder + 1) * (Norder + 2) / 2)
    
    # Inicializa a diagonal do filtro com 1.0 (sem atenuação por padrão)
    filterdiag = np.ones(Np)
    
    # Obtém o épsilon da máquina (menor número representável) e calcula alpha
    eps = np.finfo(float).eps
    alpha = -np.log(eps)
    
    # Constrói o filtro exponencial no espaço modal
    sk = 0
    for i in range(Norder + 1):
        for j in range(Norder - i + 1):
            # Se a soma dos graus do modo ultrapassar o limite de corte (Nc)
            if (i + j) >= Nc:
                # Aplica a atenuação exponencial
                filterdiag[sk] = np.exp(-alpha * (((i + j - Nc) / (Norder - Nc)) ** sp))
            sk += 1
            
    # Aplica o filtro transformando para o espaço modal e voltando para o nodal
    # F = V * diag * V^(-1)
    F = V @ np.diag(filterdiag) @ invV
    
    return F

def Lift2D(N, Np, Nfaces, Nfp, r, s, Fmask, V):
    """
    Calcula o termo de 'lift' (elevação) da superfície para o volume 
    para a formulação Discontinuous Galerkin.
    
    Assume que a função Vandermonde1D está disponível no escopo.
    """
    # Inicializa a matriz Emat com zeros
    Emat = np.zeros((Np, Nfaces * Nfp))
    
    # Garante que Fmask seja do tipo inteiro para usá-la como índice
    Fmask = np.asarray(Fmask, dtype=int)

    # --- Face 1 ---
    # Python usa índice 0 para a primeira coluna
    faceR_1 = r[Fmask[:, 0]]
    V1D_1 = op2D.Vandermonde1D(N, faceR_1)
    
    # massEdge = inv(V1D * V1D.T)
    massEdge1 = np.linalg.inv(V1D_1 @ V1D_1.T)
    
    # Preenche a primeira fatia de colunas: 0 até Nfp-1
    Emat[Fmask[:, 0], 0:Nfp] = massEdge1

    # --- Face 2 ---
    faceR_2 = r[Fmask[:, 1]]
    V1D_2 = op2D.Vandermonde1D(N, faceR_2)
    massEdge2 = np.linalg.inv(V1D_2 @ V1D_2.T)
    
    # Preenche a segunda fatia: Nfp até 2*Nfp-1
    Emat[Fmask[:, 1], Nfp:2*Nfp] = massEdge2

    # --- Face 3 ---
    # Atenção: a face 3 usa a coordenada 's' em vez de 'r'
    faceS_3 = s[Fmask[:, 2]]
    V1D_3 = op2D.Vandermonde1D(N, faceS_3)
    massEdge3 = np.linalg.inv(V1D_3 @ V1D_3.T)
    
    # Preenche a terceira fatia: 2*Nfp até 3*Nfp-1
    Emat[Fmask[:, 2], 2*Nfp:3*Nfp] = massEdge3

    # --- Cálculo Final do LIFT ---
    # Multiplicação matricial: LIFT = V * (V' * Emat)
    LIFT = V @ (V.T @ Emat)

    return LIFT

def GeometricFactors2D(x, y, Dr, Ds):
    """
    Calcula os elementos métricos (fatores geométricos) para os 
    mapeamentos locais de cada elemento da malha.
    
    Parâmetros:
    x, y  : Matrizes com as coordenadas físicas dos nós.
    Dr, Ds: Matrizes de diferenciação no elemento de referência.
    """
    # 1. Derivadas parciais das coordenadas físicas em relação a (r, s)
    # Aqui usamos o operador '@' porque é uma multiplicação matricial 
    # do operador de diferenciação (Np x Np) pelas coordenadas (Np x K)
    xr = Dr @ x
    xs = Ds @ x
    yr = Dr @ y
    ys = Ds @ y
    
    # 2. O Determinante Jacobiano da transformação (Multiplicação elemento a elemento)
    J = -xs * yr + xr * ys
    
    # 3. Os fatores métricos do mapeamento inverso (Divisão elemento a elemento)
    rx =  ys / J
    sx = -yr / J
    ry = -xs / J
    sy =  xr / J
    
    return rx, sx, ry, sy, J 

def Normals2D(x, y, Dr, Ds, Fmask, Nfp, K):
    """
    Calcula as normais apontando para fora nas faces dos elementos 
    e os Jacobianos de superfície.
    
    Parâmetros:
    x, y   : Matrizes com as coordenadas físicas dos nós.
    Dr, Ds : Matrizes de diferenciação no elemento de referência.
    Fmask  : Array com os índices dos nós que pertencem às faces.
    Nfp    : Número de nós por face.
    K      : Número total de elementos (triângulos) na malha.
    """
    # 1. Calcula as derivadas volumétricas e métricas
    xr = Dr @ x
    yr = Dr @ y
    xs = Ds @ x
    ys = Ds @ y
    
    # 2. Garante que Fmask é um vetor 1D de inteiros para indexação correta
    # (Usamos order='F' para replicar o comportamento de achatamento em coluna do MATLAB)
    Fmask_flat = np.asarray(Fmask, dtype=int).flatten(order='F')
    
    # 3. Interpola os fatores geométricos volumétricos para os nós das faces
    fxr = xr[Fmask_flat, :]
    fxs = xs[Fmask_flat, :]
    fyr = yr[Fmask_flat, :]
    fys = ys[Fmask_flat, :]
    
    # 4. Constrói as normais não-normalizadas
    nx = np.zeros((3 * Nfp, K))
    ny = np.zeros((3 * Nfp, K))
    
    # Face 1 (Índices de 0 até Nfp-1)
    slice1 = slice(0, Nfp)
    nx[slice1, :] =  fyr[slice1, :]
    ny[slice1, :] = -fxr[slice1, :]
    
    # Face 2 (Índices de Nfp até 2*Nfp-1)
    slice2 = slice(Nfp, 2 * Nfp)
    nx[slice2, :] =  fys[slice2, :] - fyr[slice2, :]
    ny[slice2, :] = -fxs[slice2, :] + fxr[slice2, :]
    
    # Face 3 (Índices de 2*Nfp até 3*Nfp-1)
    slice3 = slice(2 * Nfp, 3 * Nfp)
    nx[slice3, :] = -fys[slice3, :]
    ny[slice3, :] =  fxs[slice3, :]
    
    # 5. Normalização e cálculo do Jacobiano de Superfície (sJ)
    sJ = np.sqrt(nx * nx + ny * ny)
    
    nx = nx / sJ
    ny = ny / sJ
    
    return nx, ny, sJ

def Connect2D(EToV):
    """
    Constrói as matrizes de conectividade global (EToE e EToF) para a malha,
    baseado na matriz de entrada padrão EToV (Elementos para Vértices).
    
    Parâmetros:
    EToV : Matriz (K x 3) contendo os índices (0-based) dos nós de cada elemento.
    """
    Nfaces = 3
    K = EToV.shape[0]
    
    # Encontra o número total de vértices (como é 0-based, o Nv é o valor máximo + 1)
    Nv = np.max(EToV) + 1
    
    TotalFaces = Nfaces * K
    
    # Lista de conexões locais (face para vértice local - 0-based)
    # Face 0 conecta os nós locais 0 e 1; Face 1 os nós 1 e 2; Face 2 os nós 0 e 2
    vn = np.array([[0, 1], [1, 2], [0, 2]])
    
    # Constrói a matriz esparsa global Face -> Nó
    # Usamos o formato LIL (List of Lists) que é muito rápido para ser construído em loops
    SpFToV = sp.lil_matrix((TotalFaces, Nv))
    
    sk = 0
    for k in range(K):
        for face in range(Nfaces):
            # Obtém os vértices globais que compõem esta face
            nodes = EToV[k, vn[face, :]]
            SpFToV[sk, nodes] = 1
            sk += 1
            
    # Converte para CSR (Compressed Sparse Row) para multiplicação matricial ultra-rápida
    SpFToV = SpFToV.tocsr()
    
    # Constrói a matriz esparsa global Face -> Face
    # A mágica acontece aqui: Produto interno da matriz pela sua transposta
    SpFToF = SpFToV @ SpFToV.T - 2 * sp.eye(TotalFaces)
    
    # Encontra as conexões completas face a face (Onde o produto é 2, as faces compartilham 2 nós)
    # Convertendo para COO (Coordinate format) fica fácil extrair os índices row e col
    SpFToF = SpFToF.tocoo()
    matches = (SpFToF.data == 2)
    faces1 = SpFToF.row[matches]
    faces2 = SpFToF.col[matches]
    
    # Converte o número global da face para o número do elemento e número local da face
    # Graças ao 0-based do Python, a matemática fica perfeitamente limpa
    element1 = faces1 // Nfaces
    face1 = faces1 % Nfaces
    
    element2 = faces2 // Nfaces
    face2 = faces2 % Nfaces
    
    # Inicializa as matrizes EToE e EToF (Condição padrão: o elemento é vizinho dele mesmo nas bordas)
    EToE = np.tile(np.arange(K)[:, np.newaxis], (1, Nfaces))
    EToF = np.tile(np.arange(Nfaces)[np.newaxis, :], (K, 1))
    
    # Substitui os vizinhos padrão pelas conexões reais encontradas
    EToE[element1, face1] = element2
    EToF[element1, face1] = face2
    
    return EToE, EToF

def BuildMaps2D(x, y, Fmask, EToV, EToE, EToF, VX, VY, K, Np, Nfp, Nfaces, NODETOL=1e-8):
    """
    Constrói as tabelas de conectividade e de fronteira para a malha.
    
    Parâmetros:
    x, y   : Coordenadas nodais (Np, K)
    Fmask  : Máscara dos nós das faces
    EToV, EToE, EToF : Matrizes de conectividade dos elementos
    VX, VY : Coordenadas dos vértices da malha
    NODETOL: Tolerância numérica para encontrar nós correspondentes
    """
    # 1. Numera os nós do volume consecutivamente (0 até K*Np - 1)
    # order='F' garante a mesma ordenação por colunas do MATLAB
    nodeids = np.arange(K * Np).reshape((Np, K), order='F')
    
    vmapM = np.zeros((Nfp, Nfaces, K), dtype=int)
    vmapP = np.zeros((Nfp, Nfaces, K), dtype=int)
    
    # mapM contém índices lineares de 0 a (K*Nfp*Nfaces - 1)
    mapM = np.arange(K * Nfp * Nfaces).reshape((Nfp, Nfaces, K), order='F')
    mapP = np.copy(mapM)
    
    # Garante que Fmask é um array unidimensional (se necessário nas chamadas)
    Fmask = np.asarray(Fmask, dtype=int)
    
    # 2. Encontra os índices dos nós das faces em relação à ordem dos nós do volume
    for k1 in range(K):
        for f1 in range(Nfaces):
            vmapM[:, f1, k1] = nodeids[Fmask[:, f1], k1]
            
    # 3. Conecta as faces vizinhas
    for k1 in range(K):
        for f1 in range(Nfaces):
            # Encontra o vizinho usando as matrizes da Connect2D
            k2 = EToE[k1, f1]
            f2 = EToF[k1, f1]
            
            # Comprimento de referência da aresta
            v1 = EToV[k1, f1]
            v2 = EToV[k1, (f1 + 1) % Nfaces] # Substituto do 1+mod(f1, Nfaces)
            refd = np.sqrt((VX[v1] - VX[v2])**2 + (VY[v1] - VY[v2])**2)
            
            # Extrai os números dos nós volumétricos (local e vizinho)
            vidM = vmapM[:, f1, k1]
            vidP = vmapM[:, f2, k2]
            
            # Pega as coordenadas físicas desses nós
            x1 = x.flatten(order='F')[vidM]
            y1 = y.flatten(order='F')[vidM]
            x2 = x.flatten(order='F')[vidP]
            y2 = y.flatten(order='F')[vidP]
            
            # 4. Calcula a matriz de distância usando Broadcasting do NumPy
            # Substitui a lógica de x1*one e x2'*one do MATLAB
            D = (x1[:, np.newaxis] - x2[np.newaxis, :])**2 + \
                (y1[:, np.newaxis] - y2[np.newaxis, :])**2
            
            # Encontra quais nós estão praticamente na mesma posição geométrica
            idM, idP = np.where(np.sqrt(np.abs(D)) < NODETOL * refd)
            
            # Atualiza os mapas Plus (Vizinhos)
            vmapP[idM, f1, k1] = vidP[idP]
            # O cálculo do índice no Python usa base zero!
            mapP[idM, f1, k1] = idP + f2 * Nfp + k2 * Nfaces * Nfp

    # 5. Achata os arrays para vetores 1D (usando ordem de coluna do MATLAB)
    vmapM = vmapM.flatten(order='F')
    vmapP = vmapP.flatten(order='F')
    mapM = mapM.flatten(order='F')
    mapP = mapP.flatten(order='F')
    
    # 6. Cria a lista de nós de contorno (Boundary)
    # Na fronteira, o elemento "pensa" que é vizinho dele mesmo (vmapP == vmapM)
    mapB = np.where(vmapP == vmapM)[0]
    vmapB = vmapM[mapB]
    
    return mapM, mapP, vmapM, vmapP, vmapB, mapB