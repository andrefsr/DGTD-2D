import numpy as np
import matplotlib.pyplot as plt
import operadores2D as op2D
import Startup2D as start
import operadores2D as op2D

import meshio

def LerMalhaGmsh2D(filename):
    """
    Lê um arquivo de malha do Gmsh (.msh) e extrai os arrays estruturais 
    necessários para o solver de Galerkin Descontínuo.
    """
    print(f"Lendo malha: {filename}...")
    
    # O meshio faz todo o trabalho pesado de ler o arquivo
    mesh = meshio.read(filename)
    
    # 1. Coordenadas dos Vértices (VX, VY)
    # mesh.points retorna uma matriz (Nv, 3) com x, y, z. Pegamos apenas x e y.
    VX = mesh.points[:, 0]
    VY = mesh.points[:, 1]
    
    # 2. Número total de vértices
    Nv = len(VX)
    
    # 3. Conectividade Elemento-Vértice (EToV)
    # O meshio guarda os blocos de elementos em um dicionário. 
    # Queremos apenas os triângulos para o nosso solver 2D.
    if "triangle" not in mesh.cells_dict:
        raise ValueError("A malha não contém triângulos 2D!")
        
    EToV = mesh.cells_dict["triangle"]
    
    # 4. Número total de elementos (triângulos)
    K = EToV.shape[0]
    
    print(f"Malha carregada com sucesso: {K} elementos (triângulos), {Nv} vértices.")
    
    return Nv, VX, VY, K, EToV

def CorrigirOrientacaoMalha(VX, VY, EToV):
    """
    Garante que todos os triângulos estejam no sentido anti-horário.
    """
    # Coordenadas dos 3 vértices de todos os triângulos
    x0, y0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    x1, y1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    x2, y2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    
    # Produto vetorial para achar a área sinalizada
    area_vec = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    
    # Se a área for negativa, o triângulo está no sentido horário
    bad_tris = area_vec < 0
    num_bad = np.sum(bad_tris)
    
    if num_bad > 0:
        print(f"Corrigindo a orientação de {num_bad} triângulos...")
        # Troca as colunas 1 e 2 de lugar para inverter o sentido
        temp = EToV[bad_tris, 1].copy()
        EToV[bad_tris, 1] = EToV[bad_tris, 2]
        EToV[bad_tris, 2] = temp
        
    return EToV

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


def MaxwellRHS2D(Hx, Hy, Ez, vmapM, vmapP, mapB, vmapB, nx, ny, Fscale, LIFT, Dr, Ds, rx, sx, ry, sy):
    """
    Avalia o fluxo do Lado Direito (RHS) das Equações de Maxwell 2D na forma TM.
    """
    # 1. Achata os campos temporariamente para indexação linear (Padrão MATLAB: order='F')
    Hx_flat = Hx.flatten(order='F')
    Hy_flat = Hy.flatten(order='F')
    Ez_flat = Ez.flatten(order='F')

    # 2. Define a diferença dos campos nas faces (O "Salto" na descontinuidade)
    # dU = U_local - U_vizinho
    dHx = Hx_flat[vmapM] - Hx_flat[vmapP]
    dHy = Hy_flat[vmapM] - Hy_flat[vmapP]
    dEz = Ez_flat[vmapM] - Ez_flat[vmapP]

    # 3. Impõe as Condições de Contorno Reflexivas (Parede / PEC)
    # Para refletir perfeitamente, o campo elétrico tangencial Ez externo é espelhado negativamente (Ez+ = -Ez-)
    # Isso faz com que dEz = Ez_local - (-Ez_local) = 2 * Ez_local
    dHx[mapB] = 0.0
    dHy[mapB] = 0.0
    dEz[mapB] = 2.0 * Ez_flat[vmapB]

    # 4. Avalia os Fluxos Upwind (Fluxos Numéricos na Fronteira)
    alpha = 1.0
    nx_flat = nx.flatten(order='F')
    ny_flat = ny.flatten(order='F')

    ndotdH = nx_flat * dHx + ny_flat * dHy

    fluxHx =  ny_flat * dEz + alpha * (ndotdH * nx_flat - dHx)
    fluxHy = -nx_flat * dEz + alpha * (ndotdH * ny_flat - dHy)
    fluxEz = -nx_flat * dHy + ny_flat * dHx - alpha * dEz

    # 5. Derivadas Locais dos campos dentro do volume de cada triângulo
    # (Usando as funções traduzidas anteriormente)
    Ezx, Ezy = op2D.Grad2D(Ez, Dr, Ds, rx, sx, ry, sy)
    CuHx, CuHy, CuHz = op2D.Curl2D(Hx, Hy, None, Dr, Ds, rx, sx, ry, sy)

    # 6. Reorganiza os fluxos para o formato matricial adequado para o LIFT
    # Extraímos as dimensões (Nfaces*Nfp, K) a partir do formato da normal 'nx'
    NfacesNfp, K = nx.shape
    Fscale_flat = Fscale.flatten(order='F')

    fluxHx_scaled = (Fscale_flat * fluxHx).reshape((NfacesNfp, K), order='F')
    fluxHy_scaled = (Fscale_flat * fluxHy).reshape((NfacesNfp, K), order='F')
    fluxEz_scaled = (Fscale_flat * fluxEz).reshape((NfacesNfp, K), order='F')

    # 7. Computa o Lado Direito (RHS) das EDPs
    # Equação final: RHS = Volume(Rotacional/Gradiente) + Superfície(LIFT * Fluxo)
    rhsHx = -Ezy + (LIFT @ fluxHx_scaled) / 2.0
    rhsHy =  Ezx + (LIFT @ fluxHy_scaled) / 2.0
    rhsEz = CuHz + (LIFT @ fluxEz_scaled) / 2.0

    return rhsHx, rhsHy, rhsEz

def Maxwell2D(Hx, Hy, Ez, FinalTime, N, Np, K, rLGL, dtscale, rk4a, rk4b, 
              vmapM, vmapP, mapB, vmapB, nx, ny, Fscale, LIFT, Dr, Ds, rx, sx, ry, sy):
    """
    Integra as equações de Maxwell no modo TM no tempo até FinalTime,
    começando com os campos iniciais Hx, Hy e Ez.
    """
    time = 0.0
    
    # Armazenamento de resíduos do Runge-Kutta (Low-Storage)
    resHx = np.zeros((Np, K))
    resHy = np.zeros((Np, K))
    resEz = np.zeros((Np, K))
    
    # Desempacota os nós e os pesos separadamente
    nodes, weights = op2D.JacobiGQ(0, 0, N)

    # Calcula a distância mínima usando apenas o array de nós
    rmin = np.abs(nodes[0] - nodes[1])

    # Computa o tamanho do passo no tempo (dt) com base na condição CFL
    # rLGL representa os nós de quadratura (ex: gerados por JacobiGQ)
    #rmin = np.abs(rLGL[0] - rLGL[1])
    dt = np.min(dtscale) * rmin * (2.0 / 3.0)*0.2
    
    print(f"Iniciando simulação. dt calculado = {dt:.6e}")
    
    # Loop principal de avanço no tempo
    while time < FinalTime:
        
        # Garante que o último passo pare exatamente no FinalTime
        if (time + dt) > FinalTime:
            dt = FinalTime - time
            
        # Loop interno do Runge-Kutta de 5 estágios
        for INTRK in range(5):
            
            # 1. Calcula o lado direito (RHS) das equações de Maxwell
            rhsHx, rhsHy, rhsEz = MaxwellRHS2D(
                Hx, Hy, Ez, vmapM, vmapP, mapB, vmapB, 
                nx, ny, Fscale, LIFT, Dr, Ds, rx, sx, ry, sy
            )
            
            # 2. Inicia e incrementa os resíduos de integração
            resHx = rk4a[INTRK] * resHx + dt * rhsHx
            resHy = rk4a[INTRK] * resHy + dt * rhsHy
            resEz = rk4a[INTRK] * resEz + dt * rhsEz
            
            # 3. Atualiza os campos físicos
            Hx = Hx + rk4b[INTRK] * resHx
            Hy = Hy + rk4b[INTRK] * resHy
            Ez = Ez + rk4b[INTRK] * resEz
            
        # Incrementa o relógio da simulação
        time += dt
        
        # (Opcional) Adicione um print a cada X iterações para monitorar o progresso
        # print(f"Tempo atual: {time:.5e} / {FinalTime:.5e}")
        
    return Hx, Hy, Ez, time

# Importe todas as suas funções aqui (StartUp2D, Maxwell2D, dtscale2D, etc.)
# Importe também os coeficientes do Runge-Kutta (rk4a, rk4b)

# ==============================================================================
# Script Principal para resolver as Equações de Maxwell 2D no vácuo (Modo TM)
# ==============================================================================

# 1. Ordem polinomial usada para a aproximação espectral
N = 5

# 2. Leitura da Malha
# (Nota: Assumindo que você tem uma função para ler o arquivo .neu)
#Nv, VX, VY, K, EToV = MeshReaderGambit2D('Maxwell025.neu')
# 2. Leitura da Malha (Agora usando Gmsh + Meshio!)
#Nv, VX, VY, K, EToV = LerMalhaGmsh2D('untitled.msh')

Nv, VX, VY, K, EToV = LerMalhaGmsh2D('untitled.msh')
#EToV = CorrigirOrientacaoMalha(VX, VY, EToV) # <--- Adicione isto!

# 3. Inicializa o solver, construindo a malha, métricas e conectividades
print(f"Inicializando Galerkin Descontínuo (Ordem N={N})...")
mesh = start.StartUp2D(N, VX, VY, EToV)

# Extraindo matrizes do dicionário para facilitar a leitura
x = mesh['x']
y = mesh['y']
Np = mesh['Np']

# 4. Condições Iniciais (Modo de Cavidade Ressonante)
mmode = 1
nmode = 1

# O campo elétrico Ez começa como uma onda estacionária
Ez = np.sin(mmode * np.pi * x) * np.sin(nmode * np.pi * y)

# Os campos magnéticos começam zerados
Hx = np.zeros((Np, K))
Hy = np.zeros((Np, K))

# 5. Parâmetros de Tempo e Segurança (CFL)
FinalTime = 1.0

# Gera os nós de integração 1D para o limite do passo de tempo
rLGL = op2D.JacobiGQ(0, 0, N) 

# Calcula a escala geométrica restritiva de cada triângulo da malha
dt_scale = start.dtscale2D(x, y, mesh['r'], mesh['s'])

# 6. Resolve o Problema no Tempo
print(f"Iniciando integração no tempo até FinalTime = {FinalTime}...")

Hx, Hy, Ez, time = Maxwell2D(
    Hx, Hy, Ez, FinalTime, N, Np, K, 
    rLGL, dt_scale, rk4a, rk4b, 
    mesh['vmapM'], mesh['vmapP'], mesh['mapB'], mesh['vmapB'], 
    mesh['nx'], mesh['ny'], mesh['Fscale'], mesh['LIFT'], 
    mesh['Dr'], mesh['Ds'], mesh['rx'], mesh['sx'], mesh['ry'], mesh['sy']
)

print(f"Simulação concluída com sucesso no tempo t = {time:.4f}")

# ==============================================================================
# Fim do Solver. A partir daqui, você pode plotar os resultados ou exportá-los!
# ==============================================================================

plt.figure()
plt.grid(alpha=0.3)
plt.imshow(Ez, 
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower',cmap= 'RdGy'
            )
plt.show()