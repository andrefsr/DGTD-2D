import numpy as np
import setup as stp
import Maxwell2D as Max
import operators2D as op2D
import mesh_reader as msh
import matplotlib.pyplot as plt

### Driver Script for solving the 2D vacuum Maxwell's equations on TM form

N = 4

VX, VY, EToV, BCTags = msh.MeshReader2D('cavidade quadrada.msh')

# =====================================================================
# BLOCO DE CORREÇÃO: Força todos os triângulos para o sentido Anti-Horário
# =====================================================================
# Extrai as coordenadas x e y dos 3 vértices de cada triângulo
x1, y1 = VX[EToV[:, 0]], VY[EToV[:, 0]]
x2, y2 = VX[EToV[:, 1]], VY[EToV[:, 1]]
x3, y3 = VX[EToV[:, 2]], VY[EToV[:, 2]]

# Calcula o Determinante (Área geométrica direcional)
J_geom = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

# Encontra os índices dos triângulos que estão "do avesso" (J < 0)
triangulos_invertidos = np.where(J_geom < 0)[0]

if len(triangulos_invertidos) > 0:
    #print(f"⚠️ Corrigindo {len(triangulos_invertidos)} triângulos invertidos pelo Gmsh...")
    # Para inverter o sentido, basta trocar os nós 2 e 3 de lugar!
    temp = EToV[triangulos_invertidos, 1].copy()
    EToV[triangulos_invertidos, 1] = EToV[triangulos_invertidos, 2]
    EToV[triangulos_invertidos, 2] = temp
#else:
    #print("✅ Todos os triângulos já estão no sentido correto.")
# =====================================================================

malha = stp.StartUp2D(N,EToV,VX,VY)

### Condições iniciais
Ez = np.sin(np.pi*malha.x)*np.sin(np.pi*malha.y)
Hx = np.zeros((malha.Np,malha.K))
Hy = np.zeros((malha.Np,malha.K))

FinalTime = 10
Hx, Hy, Ez = Max.Maxwell2D(Hx,Hy,Ez,FinalTime,malha)


###########################################################################################################################


# 1. Achata as matrizes para vetores 1D
x_plot = malha.x.flatten(order='F')
y_plot = malha.y.flatten(order='F')
Ez_plot = Ez.flatten(order='F')

# 2. Configura a janela do gráfico
plt.figure(figsize=(8, 6))
plt.title(f'Campo Elétrico (Ez) em t = {FinalTime}')

# 3. Plota o campo (cmap='seismic' é ótimo para ondas: Azul negativo, Vermelho positivo)
grafico = plt.tricontourf(x_plot, y_plot, Ez_plot, levels=100, cmap='seismic')
plt.colorbar(grafico, label='Amplitude Ez')

# (Opcional) Descomente a linha abaixo se quiser ver a malha de triângulos por cima do campo!
plt.plot(VX[EToV].T, VY[EToV].T, color='k', linewidth=0.5, alpha=0.3)

# 4. Formatação
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal') # Força o quadrado a não ficar esticado
plt.tight_layout()

# 5. Mostra na tela
plt.show()

# 1. Achatamos as coordenadas globais de todos os nós de todos os triângulos
x_nos = malha.x.flatten(order='F')
y_nos = malha.y.flatten(order='F')

# 2. Preparamos a figura
plt.figure(figsize=(8, 8))
plt.title(f'Distribuição dos Nós no Nodal DG (Polinômio N = {N})')

# 3. Desenhamos o "aramado" (as arestas da malha) em cinza claro ao fundo
# Note que usamos VX, VY e EToV (a malha bruta do Gmsh) para desenhar as linhas
plt.triplot(VX, VY, EToV, color='gray', linewidth=0.8, alpha=0.5, label='Arestas dos Triângulos')

# 4. Desenhamos os nós como pontinhos vermelhos por cima
plt.plot(x_nos, y_nos, 'o', markersize=3, color='red', label='Nós de Interpolação')

# 5. Formatação
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal') # Garante que o quadrado não fique distorcido
plt.legend(loc='upper right')

plt.show()