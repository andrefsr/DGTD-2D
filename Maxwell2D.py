

def MaxwellRhs2D(Hx, Hy, Ez, malha, op2D):
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
    
    return rhsHx, rhsHy, rhsEz

