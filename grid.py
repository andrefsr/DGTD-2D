import numpy as np
import Nodes
import operators2D as op2D

def GeometricFactors2D(x,y,Dr,Ds):
    '''Compute the metric elements for the local mappings of the elements'''

    xr = Dr @ x
    xs = Ds @ x
    yr = Dr @ y
    ys = Ds @ y
    J = - xs * yr + xr * ys

    rx = ys/J
    sx = -yr/J
    ry = -xs/J
    sy = xr/J

    return rx, sx, ry, sy, J

def Normals2D(x, y, Dr, Ds, Fmask, N):
    """
    Calcula normais e Jacobianos de superfície.
    """
    K = x.shape[1]
    Nfp = N + 1 # Pode calcular o Nfp aqui dentro sem problemas!
    
    xr = Dr @ x; yr = Dr @ y
    xs = Ds @ x; ys = Ds @ y
    
    # Fmask já precisa vir calculada com (r,s) lá do script principal!
    Fmask_flat = Fmask.flatten(order='F')
    
    fxr = xr[Fmask_flat, :]
    fxs = xs[Fmask_flat, :]
    fyr = yr[Fmask_flat, :]
    fys = ys[Fmask_flat, :]
    
    nx = np.zeros((3 * Nfp, K))
    ny = np.zeros((3 * Nfp, K))
    
    fid1 = slice(0, Nfp)
    fid2 = slice(Nfp, 2 * Nfp)
    fid3 = slice(2 * Nfp, 3 * Nfp)
    
    nx[fid1, :] =  fyr[fid1, :]
    ny[fid1, :] = -fxr[fid1, :]
    
    nx[fid2, :] =  fys[fid2, :] - fyr[fid2, :]
    ny[fid2, :] = -fxs[fid2, :] + fxr[fid2, :]
    
    nx[fid3, :] = -fys[fid3, :]
    ny[fid3, :] =  fxs[fid3, :]
    
    sJ = np.sqrt(nx**2 + ny**2)
    nx = nx / sJ
    ny = ny / sJ
    
    return nx, ny, sJ

def Connect2D(EToV): # ?????
    """
    Constrói as matrizes de conectividade Elemento-Elemento (EToE) 
    e Elemento-Face (EToF) a partir da topologia EToV.
    """
    K = EToV.shape[0] # Número total de elementos
    Nfaces = 3
    
    # Inicializa assumindo que todo mundo é vizinho de si mesmo (Bordas do domínio)
    EToE = np.tile(np.arange(K)[:, None], (1, Nfaces))
    EToF = np.tile(np.arange(Nfaces)[None, :], (K, 1))
    
    # Dicionário para rastrear as faces. 
    # Chave = "Assinatura" dos vértices, Valor = (Elemento, Face)
    face_map = {}
    
    # A convenção padrão do Hesthaven para as arestas locais:
    # Face 0: liga vértice 0 e 1
    # Face 1: liga vértice 1 e 2
    # Face 2: liga vértice 2 e 0
    face_vertices = [(0, 1), (1, 2), (2, 0)]
    
    # Percorre todos os triângulos e suas 3 faces
    for k in range(K):
        for f in range(Nfaces):
            # Pega os IDs locais dos vértices que formam a face
            v1_local, v2_local = face_vertices[f]
            
            # Pega os IDs globais desses vértices na matriz EToV
            node1 = EToV[k, v1_local]
            node2 = EToV[k, v2_local]
            
            # Cria uma "assinatura" ordenando os números para que 
            # (A, B) seja igual a (B, A)
            signature = tuple(sorted([node1, node2]))
            
            if signature in face_map:
                # OPA! Alguém já registrou essa face. Achamos um vizinho!
                vizinho_k, vizinho_f = face_map[signature]
                
                # Conecta o elemento atual ao vizinho
                EToE[k, f] = vizinho_k
                EToF[k, f] = vizinho_f
                
                # Conecta o vizinho ao elemento atual (via de mão dupla)
                EToE[vizinho_k, vizinho_f] = k
                EToF[vizinho_k, vizinho_f] = f
            else:
                # Primeira vez vendo essa face. Salva no mapa esperando o vizinho.
                face_map[signature] = (k, f)
                
    return EToE, EToF 

def BuildMaps2D(x, y, Fmask, EToV, EToE, EToF, VX, VY, NODETOL=1e-10):
    """
    Constrói as tabelas de conectividade e contorno para a malha nodal.
    """
    Np, K = x.shape
    Nfaces = 3
    Nfp = Fmask.shape[0] // Nfaces # Nós por face
    
    # 1. Numera os nós de volume consecutivamente (Lido como Fortran)
    nodeids = np.reshape(np.arange(K * Np), (Np, K), order='F')
    
    # Inicializa os mapas com zeros
    vmapM = np.zeros((Nfp, Nfaces, K), dtype=int)
    vmapP = np.zeros((Nfp, Nfaces, K), dtype=int)
    
    # mapM é apenas uma lista sequencial de todos os nós de face
    mapM = np.arange(K * Nfp * Nfaces)
    mapP = np.reshape(np.copy(mapM), (Nfp, Nfaces, K), order='F')
    
    # 2. Preenche vmapM (os nós internos vistos da borda)
    Fmask_2d = np.reshape(Fmask, (Nfp, Nfaces), order='F')
    for k1 in range(K):
        for f1 in range(Nfaces):
            vmapM[:, f1, k1] = nodeids[Fmask_2d[:, f1], k1]
            
    # 3. O Loop Principal de Conexão dos Vizinhos
    for k1 in range(K):
        for f1 in range(Nfaces):
            # Encontra o vizinho k2 e a face do vizinho f2
            k2 = EToE[k1, f1]
            f2 = EToF[k1, f1]
            
            # Comprimento de referência da aresta geométrica (usando vértices globais)
            v1 = EToV[k1, f1]
            v2 = EToV[k1, (f1 + 1) % Nfaces] # O mod do MATLAB virou %
            refd = np.sqrt((VX[v1] - VX[v2])**2 + (VY[v1] - VY[v2])**2)
            
            # IDs de volume dos nós da face atual (Minus) e da face vizinha (Plus)
            vidM = vmapM[:, f1, k1]
            vidP = vmapM[:, f2, k2]
            
            # Coordenadas físicas (x,y) desses nós
            x1, y1 = x.flatten(order='F')[vidM], y.flatten(order='F')[vidM]
            x2, y2 = x.flatten(order='F')[vidP], y.flatten(order='F')[vidP]
            
            # Cálculo da Matriz de Distância D entre todos os nós da Face 1 vs Face 2
            # Usa broadcasting do NumPy (x1[:, None] transforma num vetor coluna)
            D = (x1[:, None] - x2[None, :])**2 + (y1[:, None] - y2[None, :])**2
            
            # Encontra quais nós colidem (distância menor que a tolerância geométrica)
            idM, idP = np.where(np.sqrt(np.abs(D)) < NODETOL * refd)
            
            # Preenche os mapas PLUS (com as informações de k2 e f2 achadas)
            vmapP[idM, f1, k1] = vidP[idP]
            # Cálculo do índice linear (cuidado: matemática de base 0 no Python)
            mapP[idM, f1, k1] = idP + (f2 * Nfp) + (k2 * Nfaces * Nfp)
            
    # 4. Achata as matrizes em vetores 1D (usando ordem Fortran para manter compatibilidade)
    vmapP = vmapP.flatten(order='F')
    vmapM = vmapM.flatten(order='F')
    mapP = mapP.flatten(order='F')
    
    # 5. Lista de Contorno (Boundary Nodes)
    # Qualquer nó onde o Plus aponta para o próprio Minus significa que não tem vizinho!
    mapB = np.where(vmapP == vmapM)[0]
    vmapB = vmapM[mapB]
    
    return mapM, mapP, vmapM, vmapP, vmapB, mapB


