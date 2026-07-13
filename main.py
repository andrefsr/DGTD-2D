import numpy as np
import setup as stp
import Maxwell2D as Max
import operators2D as op2D
import mesh_reader as msh

### Driver Script for solving the 2D vacuum Maxwell's equations on TM form

N = 10

VX, VY, EToV, BCTags = msh.MeshReader2D('cavidade quadrada.msh')

malha = stp.StartUp2D(N,EToV,VX,VY)

### Condições iniciais
Ez = np.sin(np.pi*malha.x)*np.sin(np.pi*malha.y)
Hx = np.zeros((malha.Np,malha.K))
Hy = np.zeros((malha.Np,malha.K))

FinalTime = 1
Hx, Hy, Ez = Max.Maxwell2D(Hx,Hy,Ez,FinalTime,malha)
