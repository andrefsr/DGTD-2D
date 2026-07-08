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
