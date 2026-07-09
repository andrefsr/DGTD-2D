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
