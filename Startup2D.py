import numpy as np
import matrizes as mat
import operadores2D as op2D
import triangulo_ref as tref

def StartUp2D(N, VX, VY, EToV):
    """
    Script de setup: constrói os operadores, a malha, os termos métricos 
    e as tabelas de conectividade.
    """
    # --- 1. Definição de Constantes ---
    Nfp = N + 1
    Np = int((N + 1) * (N + 2) / 2)
    Nfaces = 3
    NODETOL = 1e-12
    K = EToV.shape[0]

    # --- 2. Conjunto Nodal de Referência ---
    # Usa as funções previamente traduzidas para o triângulo
    x_eq, y_eq = tref.Nodes2D(N) 
    r, s = tref.xytors(x_eq, y_eq)

    # --- 3. Matrizes do Elemento de Referência ---
    V = mat.Vandermonde2D(N, r, s)
    invV = np.linalg.inv(V)
    MassMatrix = invV.T @ invV
    Dr, Ds = mat.Dmatrices2D(N, r, s, V) # Assume que Dmatrices2D foi definida

    # --- 4. Coordenadas Físicas de Todos os Nós (Broadcasting) ---
    va = EToV[:, 0]
    vb = EToV[:, 1]
    vc = EToV[:, 2]

    r_col = r.reshape(-1, 1)
    s_col = s.reshape(-1, 1)

    # Mapeamento afim do triângulo de referência para a malha real
    x = 0.5 * (-(r_col + s_col) * VX[va] + (1 + r_col) * VX[vb] + (1 + s_col) * VX[vc])
    y = 0.5 * (-(r_col + s_col) * VY[va] + (1 + r_col) * VY[vb] + (1 + s_col) * VY[vc])

    # --- 5. Máscara das Faces ---
    # Encontra todos os nós que repousam sobre cada aresta
    fmask1 = np.where(np.abs(s + 1) < NODETOL)[0]
    fmask2 = np.where(np.abs(r + s) < NODETOL)[0]
    fmask3 = np.where(np.abs(r + 1) < NODETOL)[0]
    Fmask = np.column_stack((fmask1, fmask2, fmask3))
    
    Fmask_flat = Fmask.flatten(order='F')
    Fx = x[Fmask_flat, :]
    Fy = y[Fmask_flat, :]

    # --- 6. Integrais de Superfície e Fatores Geométricos ---
    LIFT = mat.Lift2D(N, Np, Nfaces, Nfp, r, s, Fmask, V)
    rx, sx, ry, sy, J = mat.GeometricFactors2D(x, y, Dr, Ds)
    nx, ny, sJ = mat.Normals2D(x, y, Dr, Ds, Fmask, Nfp, K)
    
    # Fator de escala para a integração de contorno
    Fscale = sJ / J[Fmask_flat, :]

    # --- 7. Tabelas de Conectividade ---
    EToE, EToF = mat.Connect2D(EToV)
    mapM, mapP, vmapM, vmapP, vmapB, mapB = mat.BuildMaps2D(
        x, y, Fmask, EToV, EToE, EToF, VX, VY, K, Np, Nfp, Nfaces, NODETOL
    )

    # --- 8. Operadores Fracos (Integração por Partes) ---
    Vr, Vs = mat.GradVandermonde2D(N, r, s)
    
    # Substituição eficiente da divisão matricial do MATLAB
    Drw = (V @ Vr.T) @ MassMatrix
    Dsw = (V @ Vs.T) @ MassMatrix

    # --- Retorno Estruturado ---
    return {
        'Np': Np, 'Nfp': Nfp, 'Nfaces': Nfaces, 'K': K,
        'r': r, 's': s, 'x': x, 'y': y,
        'V': V, 'invV': invV, 'MassMatrix': MassMatrix,
        'Dr': Dr, 'Ds': Ds, 'Drw': Drw, 'Dsw': Dsw,
        'LIFT': LIFT, 'Fmask': Fmask, 'Fx': Fx, 'Fy': Fy,
        'rx': rx, 'sx': sx, 'ry': ry, 'sy': sy, 'J': J,
        'nx': nx, 'ny': ny, 'sJ': sJ, 'Fscale': Fscale,
        'EToE': EToE, 'EToF': EToF,
        'mapM': mapM, 'mapP': mapP, 'vmapM': vmapM, 'vmapP': vmapP,
        'vmapB': vmapB, 'mapB': mapB
    }

# ==========================================
# Coeficientes do Runge-Kutta de Baixo Armazenamento (LSRK45)
# ==========================================
rk4a = np.array([
    0.0,
    -567301805773.0  / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0
])

rk4b = np.array([
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0
])

rk4c = np.array([
    0.0,
    1432997174477.0 / 9575080441755.0,
    2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0,
    2802321613138.0 / 2924317926251.0
])

import numpy as np

def dtscale2D(x, y, r, s, NODETOL=1e-12):
    """
    Calcula o raio do círculo inscrito como comprimento característico
    da malha para determinar o passo de tempo limite (dt).
    
    Parâmetros:
    x, y : Coordenadas físicas de todos os nós (Np, K).
    r, s : Coordenadas dos nós no triângulo de referência (Np,).
    """
    # 1. Encontra os índices (no triângulo de referência) que correspondem aos 3 vértices
    # np.where retorna um array de índices. Pegamos o primeiro [0] escalar.
    vmask1 = np.where(np.abs(s + r + 2.0) < NODETOL)[0][0]
    vmask2 = np.where(np.abs(r - 1.0) < NODETOL)[0][0]
    vmask3 = np.where(np.abs(s - 1.0) < NODETOL)[0][0]
    
    # Agrupa os índices dos vértices
    vmask = [vmask1, vmask2, vmask3]
    
    # 2. Extrai as coordenadas físicas reais apenas desses 3 vértices para todos os elementos
    # vx e vy ficarão com o formato (3, K), onde K é o número de triângulos
    vx = x[vmask, :]
    vy = y[vmask, :]
    
    # 3. Calcula o comprimento das 3 arestas de cada triângulo físico
    len1 = np.sqrt((vx[0, :] - vx[1, :])**2 + (vy[0, :] - vy[1, :])**2)
    len2 = np.sqrt((vx[1, :] - vx[2, :])**2 + (vy[1, :] - vy[2, :])**2)
    len3 = np.sqrt((vx[2, :] - vx[0, :])**2 + (vy[2, :] - vy[0, :])**2)
    
    # 4. Calcula o semiperímetro
    sper = (len1 + len2 + len3) / 2.0
    
    # 5. Calcula a Área de cada triângulo usando a Fórmula de Heron
    Area = np.sqrt(sper * (sper - len1) * (sper - len2) * (sper - len3))
    
    # 6. O fator de escala é o raio do círculo inscrito (Inradius)
    # Matematicamente: Raio = Área / Semiperímetro
    dtscale = Area / sper
    
    return dtscale

import numpy as np

def BuildBCMaps2D(BCType, Nfp, vmapM):
    """
    Constrói mapas nodais especializados para vários tipos de 
    condições de contorno, especificados na matriz BCType.
    
    Parâmetros:
    BCType : Matriz (Nfaces, K) com as flags de fronteira de cada face.
    Nfp    : Número de nós por face.
    vmapM  : Mapa de nós locais (1D array) construído na BuildMaps2D.
    """
    # Constantes de Condição de Contorno (mesmas do arquivo anterior)
    BC_IN = 1
    BC_OUT = 2
    BC_WALL = 3
    BC_FAR = 4
    BC_CYL = 5
    BC_DIRICHLET = 6
    BC_NEUMAN = 7
    BC_SLIP = 8
    
    # Achata a matriz BCType mantendo a ordem das colunas do MATLAB
    bct_flat = np.asarray(BCType).flatten(order='F')
    
    # Expande a tag de cada face para todos os nós que pertencem a ela.
    # Ex: se uma face tem BC=3 e Nfp=4, isso gera [3, 3, 3, 3]
    bnodes = np.repeat(bct_flat, Nfp)
    
    # Encontra as localizações (índices nas faces e no volume) 
    # para cada tipo de condição de contorno
    
    mapI = np.where(bnodes == BC_IN)[0]
    vmapI = vmapM[mapI]
    
    mapO = np.where(bnodes == BC_OUT)[0]
    vmapO = vmapM[mapO]
    
    mapW = np.where(bnodes == BC_WALL)[0]
    vmapW = vmapM[mapW]
    
    mapF = np.where(bnodes == BC_FAR)[0]
    vmapF = vmapM[mapF]
    
    mapC = np.where(bnodes == BC_CYL)[0]
    vmapC = vmapM[mapC]
    
    mapD = np.where(bnodes == BC_DIRICHLET)[0]
    vmapD = vmapM[mapD]
    
    mapN = np.where(bnodes == BC_NEUMAN)[0]
    vmapN = vmapM[mapN]
    
    mapS = np.where(bnodes == BC_SLIP)[0]
    vmapS = vmapM[mapS]
    
    # Retorna tudo empacotado para manter o código principal limpo
    return {
        'mapI': mapI, 'vmapI': vmapI,
        'mapO': mapO, 'vmapO': vmapO,
        'mapW': mapW, 'vmapW': vmapW,
        'mapF': mapF, 'vmapF': vmapF,
        'mapC': mapC, 'vmapC': vmapC,
        'mapD': mapD, 'vmapD': vmapD,
        'mapN': mapN, 'vmapN': vmapN,
        'mapS': mapS, 'vmapS': vmapS
    }