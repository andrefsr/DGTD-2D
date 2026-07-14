import numpy as np
import aux_func as aux
import setup
import operators2D as op2D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

mu0 = 4*np.pi*10**(-7)
eps0 = 8.854*10**(-12)
c0 = 1/np.sqrt(eps0*mu0)

def MaxwellRhs2D(Hx, Hy, Ez, malha,time):
    '''Calcula o fluxo (lado direito) das equações de Maxwell 2D para o modo TM'''
    
    # 1. Achata as matrizes em 1D (ordem Fortran) para os mapas de conectividade funcionarem
    Hx_flat = Hx.flatten(order='F')
    Hy_flat = Hy.flatten(order='F')
    Ez_flat = Ez.flatten(order='F')
    
    # 2. Calcula o salto (Minus - Plus) nas faces
    dHx = Hx_flat[malha.vmapM] - Hx_flat[malha.vmapP]
    dHy = Hy_flat[malha.vmapM] - Hy_flat[malha.vmapP]
    dEz = Ez_flat[malha.vmapM] - Ez_flat[malha.vmapP]
    
    # 3. Condição de Contorno: Condutor Elétrico Perfeito (PEC)
    # Na parede (mapB), não há salto magnético, e o salto elétrico reflete perfeitamente
    dHx[malha.mapB] = 0.0
    dHy[malha.mapB] = 0.0
    dEz[malha.mapB] = 2.0 * Ez_flat[malha.vmapB]
    
    # 4. Retorna os saltos para o formato 2D (Nós_da_Face x Elementos) 
    # para podermos multiplicar ponto-a-ponto com os vetores normais
    shape_faces = (malha.Nfp * malha.Nfaces, malha.K)
    dHx = dHx.reshape(shape_faces, order='F')
    dHy = dHy.reshape(shape_faces, order='F')
    dEz = dEz.reshape(shape_faces, order='F')
    
    # 5. Fluxos de Fronteira (Upwind)
    alpha = 1.0
    ndotdH = malha.nx * dHx + malha.ny * dHy
    
    fluxHx =  malha.ny * dEz + alpha * (ndotdH * malha.nx - dHx)
    fluxHy = -malha.nx * dEz + alpha * (ndotdH * malha.ny - dHy) # Corrigido para -dHy
    fluxEz = -malha.nx * dHy + malha.ny * dHx - alpha * dEz
    
    # 6. Derivadas Locais (Operadores de Volume)
    # Agora passamos os argumentos que elas exigem
    Ezx, Ezy = op2D.Grad2D(Ez, malha.rx, malha.sx, malha.ry, malha.sy, malha.Dr, malha.Ds)
    CuHx, CuHy, CuHz = op2D.Curl2D(Hx, Hy, None, malha.rx, malha.sx, malha.ry, malha.sy, malha.Dr, malha.Ds)
    
    # 7. Montagem do RHS final: Volume + Fluxo(Borda)
    # Correção: LIFT exige multiplicação de matriz (@)
    rhsHx = -Ezy + malha.LIFT @ (malha.Fscale * fluxHx) / 2.0
    rhsHy =  Ezx + malha.LIFT @ (malha.Fscale * fluxHy) / 2.0
    rhsEz = CuHz + malha.LIFT @ (malha.Fscale * fluxEz) / 2.0
    f = 3
    rhsEz += 2*np.pi*f*np.sin(2.0 * np.pi * f * time)*np.exp(-(malha.x**2 + malha.y**2) / 0.1**2)

    return rhsHx, rhsHy, rhsEz

def Maxwell2D(Hx, Hy, Ez, FinalTime, malha):
    '''Integrate TM-mode Maxwell's until FinalTime starting with initial conditions Hx, Hy, Ez'''
    
    # 1. Matrizes do Runge-Kutta de Baixo Armazenamento (5 estágios, 4ª ordem)
    rk4a = np.array([
        0.0, 
        -567301805773.0 / 1357537059087.0, 
        -2404267990393.0 / 2016746695238.0, 
        -3550918686646.0 / 2091501179385.0, 
        -3270041081375.0 / 2362478004168.0
    ])
    
    rk4b = np.array([
        1432997174477.0 / 9575080441755.0, 
        5161836677717.0 / 13612068292357.0, 
        1720146321549.0 / 2090206949498.0, 
        3134564353537.0 / 4481467310338.0, 
        2277821191437.0 / 14882151754819.0
    ])
    
    # A matriz rk4c é usada se o seu lado direito (RHS) depender 
    # do tempo absoluto (como fontes de antena pulsantes t=time+rk4c[INTRK]*dt)
    rk4c = np.array([
        0.0, 
        1432997174477.0 / 9575080441755.0, 
        2526269341429.0 / 6820363962896.0, 
        2006345519317.0 / 3224310063776.0, 
        2802321613138.0 / 2924317926251.0
    ])

    time = 0.0
    passo = 0 # Contador para sabermos quando atualizar a tela

    # --- PREPARAÇÃO DA ANIMAÇÃO ---
    plt.ion() # Liga o modo interativo do Matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # DICA DE OURO: Criar a triangulação uma única vez antes do loop 
    # economiza MUITO processamento!
    x_flat = malha.x.flatten(order='F')
    y_flat = malha.y.flatten(order='F')
    triangulacao = mtri.Triangulation(x_flat, y_flat)
    # ------------------------------
    
    # Registradores residuais do RK (só precisamos de um para cada variável)
    resHx = np.zeros((malha.Np, malha.K))
    resHy = np.zeros((malha.Np, malha.K))
    resEz = np.zeros((malha.Np, malha.K))
    
    # 2. Cálculo do passo de tempo (CFL)
    # Cuidado: se JacobiGQ retornar (raízes, pesos), garanta que está pegando as raízes
    rLGL, _ = aux.JacobiGQ(0, 0, malha.N) 
    rmin = np.abs(rLGL[0] - rLGL[1]) 
    
    # Chamando com as variáveis corretas
    dtscale = setup.dtscale2D(malha.x, malha.y, malha.r, malha.s)
    
    # O passo de tempo básico
    CFL = 0.2
    dt = CFL*np.min(dtscale) * rmin * (2.0/3.0)
    
    mask = np.where((malha.x == 0) & (malha.y == 0))

    # 3. O Loop de Tempo Principal
    while time < FinalTime:
        
        # Trava de segurança: impede que a simulação passe do tempo final desejado
        if time + dt > FinalTime:
            dt = FinalTime - time
            
        # Loop do Runge-Kutta (Agora com 5 estágios)
        for INTRK in range(5):
            
            # (Opcional) Se sua simulação tiver um pulso que depende do tempo, 
            # o tempo local de avaliação desse pulso seria:
            t_local = time + rk4c[INTRK] * dt
            
            # Chamada do RHS com todas as dependências corretas
            rhsHx, rhsHy, rhsEz = MaxwellRhs2D(Hx, Hy, Ez, malha,t_local)
            
            # Atualiza o residual
            resHx = rk4a[INTRK] * resHx + dt * rhsHx
            resHy = rk4a[INTRK] * resHy + dt * rhsHy
            resEz = rk4a[INTRK] * resEz + dt * rhsEz
            
            # Atualiza o campo principal
            Hx = Hx + rk4b[INTRK] * resHx
            Hy = Hy + rk4b[INTRK] * resHy
            Ez = Ez + rk4b[INTRK] * resEz
        
            #Ez[mask] = np.exp(-((time - 1.8e-9*c0)**2  / (0.6e-9*c0)**2))
        # Avança o relógio
        time += dt
        passo += 1

        # --- ATUALIZAÇÃO DA TELA (A cada 20 passos) ---
        if passo % 20 == 0:
            ax.clear() # Limpa o frame antigo
            
            Ez_flat = Ez.flatten(order='F')

            tfis_ns = (time/c0)*1e9

            
            # vmin e vmax são cruciais para a escala de cores não ficar "piscando"
            ax.tricontourf(triangulacao, Ez_flat, levels=50, cmap='seismic', vmin=-1.0, vmax=1.0)
            
            ax.set_title(f'Campo Ez - Tempo: {tfis_ns:.4f} ns')
            ax.set_aspect('equal')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            
            # Pausa minúscula para o Python ter tempo de desenhar na tela
            plt.pause(0.001) 
            
    # --- FINALIZAÇÃO ---
    plt.ioff() # Desliga o modo interativo quando acabar
    plt.show() # Mantém a última tela aberta
        #print(f"Tempo atual: {time:.4e} / {FinalTime:.4e}") # Opcional: print para não ficar cego
        
    return Hx, Hy, Ez


