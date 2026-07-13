import numpy as np
import Nodes
import grid
import operators2D as op2D
from types import SimpleNamespace

def StartUp2D(N,EToV,VX,VY):
    Nfp = N + 1
    Np = int(((N+1)*(N+2))/2)
    Nfaces = 3
    NODETOL = 1e-12

    x, y = Nodes.Nodes2D(N)
    r, s = Nodes.xytors(x,y)

    V = op2D.Vandermonde2D(N,r,s)
    invV = np.linalg.inv(V)
    MassMatrix = invV.T @ invV

    Dr, Ds = op2D.Dmatrices2D(N,r,s,V)
    va = EToV[:,0].T
    vb = EToV[:,1].T
    vc = EToV[:,2].T

    r_col = r[:, None]
    s_col = s[:, None]
    
    x = 0.5 * (-(r_col + s_col) * VX[va] + (1 + r_col) * VX[vb] + (1 + s_col) * VX[vc])
    y = 0.5 * (-(r_col + s_col) * VY[va] + (1 + r_col) * VY[vb] + (1 + s_col) * VY[vc])

    Fmask = op2D.calcular_fmask(r,s,NODETOL)
    fmask_flat = Fmask.flatten(order='F')
    Fx = x[fmask_flat, :]
    Fy = y[fmask_flat, :]

    LIFT = op2D.Lift2D(N,r,s,V,Fmask)

    rx, sx, ry, sy, J = grid.GeometricFactors2D(x,y,Dr,Ds)

    nx, ny, sJ = grid.Normals2D(x,y,Dr,Ds,Fmask,N)
    Fscale = sJ/J[fmask_flat,:]

    EToE, EToF = grid.Connect2D(EToV)

    mapM, mapP, vmapM, vmapP, vmapB, mapB = grid.BuildMaps2D(x,y,Fmask,EToV,EToE,EToF,VX,VY,NODETOL)

    Vr, Vs = op2D.GradVandermonde2D(N,r,s)
    Drw = (V @ Vr.T) @ np.linalg.inv(V @ V.T)
    Dsw = (V @ Vs.T) @ np.linalg.inv(V @ V.T)

    # Empacota TODAS as variáveis da malha em um único objeto organizado
    malha = SimpleNamespace(
        N=N, Np=Np, Nfp=Nfp, Nfaces=Nfaces, K=EToV.shape[0],
        r=r, s=s, x=x, y=y,
        V=V, invV=invV, MassMatrix=MassMatrix, 
        Dr=Dr, Ds=Ds, Drw=Drw, Dsw=Dsw, LIFT=LIFT,
        rx=rx, sx=sx, ry=ry, sy=sy, J=J,
        nx=nx, ny=ny, sJ=sJ, Fscale=Fscale,
        EToE=EToE, EToF=EToF, Fmask=Fmask, Fx=Fx, Fy=Fy,
        mapM=mapM, mapP=mapP, vmapM=vmapM, vmapP=vmapP, vmapB=vmapB, mapB=mapB
    )
    
    return malha

def dtscale2D(x, y, r, s):
    """
    Calcula o raio do círculo inscrito de cada triângulo da malha 
    para ser usado na condição de estabilidade CFL (passo de tempo).
    
    x, y : Coordenadas globais dos nós da malha (Np, K)
    r, s : Coordenadas do elemento de referência (Np, )
    """

    #No DG, a velocidade máxima da onda não pode ultrapassar o tamanho do menor elemento da malha
    #  (para que a onda não pule um triângulo inteiro em um único passo de tempo).
    #  Como triângulos finos e esticados são perigosos, 
    #  o "raio do círculo inscrito" é a medida mais conservadora e segura da "menor espessura" de um triângulo!
    #  O vetor dtscale resultante terá o tamanho seguro para cada elemento individualmente

    NODETOL = 1e-12
    
    vmask1 = np.where(np.abs(s + r + 2.0) < NODETOL)[0][0]
    vmask2 = np.where(np.abs(r - 1.0) < NODETOL)[0][0]
    vmask3 = np.where(np.abs(s - 1.0) < NODETOL)[0][0]
    
    vmask = [vmask1, vmask2, vmask3]
    
    vx = x[vmask, :]
    vy = y[vmask, :]
    
    len1 = np.sqrt((vx[0, :] - vx[1, :])**2 + (vy[0, :] - vy[1, :])**2)
    len2 = np.sqrt((vx[1, :] - vx[2, :])**2 + (vy[1, :] - vy[2, :])**2)
    len3 = np.sqrt((vx[2, :] - vx[0, :])**2 + (vy[2, :] - vy[0, :])**2)
    
    # 4. Semiperímetro
    sper = (len1 + len2 + len3) / 2.0
    
    # 5. Área (Fórmula de Heron)
    Area = np.sqrt(sper * (sper - len1) * (sper - len2) * (sper - len3))
    
    # 6. Escala de tempo (Raio do círculo inscrito)
    dtscale = Area / sper
    
    return dtscale

def BuildBCMaps2D(BCType, Nfp, vmapM):
    """
    Constrói mapas nodais especializados para os vários tipos 
    de condições de contorno especificados na matriz BCType.
    
    BCType : Matriz (K, Nfaces) contendo as flags numéricas de cada face.
             (Faces internas costumam ter valor 0).
    Nfp    : Número de nós por face.
    vmapM  : Mapa global de nós (Minus), já em formato 1D (Fortran order).
    """
    
    # 1. Dicionário padrão de Condições de Contorno do Hesthaven.
    # (Se a sua malha usar números diferentes para as bordas, basta alterar aqui)
    BC_FLAGS = {
        'In': 1, 'Out': 2, 'Wall': 3, 'Far': 4,
        'Cyl': 5, 'Dirichlet': 6, 'Neuman': 7, 'Slip': 8
    }
    
    # 2. Transpõe e achata a matriz BCType para alinhar a memória
    # com a forma que o vmapM foi achatado anteriormente (Ordem Fortran)
    bct_flat = BCType.T.flatten(order='F')
    
    # 3. Repete a flag da face para todos os 'Nfp' nós pertencentes a ela.
    # Isso equivale à multiplicação de matrizes do MATLAB: ones(Nfp, 1) * bct
    bnodes = np.repeat(bct_flat, Nfp)
    
    # 4. Inicializa um dicionário vazio para guardar as listas
    bc_maps = {}
    
    # 5. Procura e agrupa os nós correspondentes a cada tipo de borda
    for name, flag in BC_FLAGS.items():
        # Encontra os índices lineares onde a flag da borda bate com a procurada
        map_idx = np.where(bnodes == flag)[0]
        
        # Se encontrou algum nó com essa condição na malha, salva no dicionário
        if len(map_idx) > 0:
            # Equivalente ao mapI, mapW, mapO, etc...
            bc_maps[f'map{name}'] = map_idx
            
            # Equivalente ao vmapI, vmapW, vmapO, etc...
            bc_maps[f'vmap{name}'] = vmapM[map_idx]
            
    # Converte o dicionário em um objeto fácil de acessar (namespace)
    return SimpleNamespace(**bc_maps)

# Chama a função
#fronteiras = BuildBCMaps2D(BCType, malha.Nfp, malha.vmapM)

# Se a sua malha tiver uma parede (Wall), você pode acessar os nós dela 
# diretamente, como um atributo do objeto:
#nos_da_parede = fronteiras.vmapWall
#indices_da_parede = fronteiras.mapWall
