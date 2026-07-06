import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

def rstoab(r, s):
    """
    Purpose : Transfer from (r,s) -> (a,b) coordinates in triangle
    """
    # Garante que as entradas sejam arrays do NumPy para permitir operações matemáticas diretas
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    
    # Inicializa o vetor 'a' com a mesma estrutura de 'r'
    a = np.zeros_like(r)
    
    # Cria uma máscara booleana onde 's' é diferente de 1
    mask = (s != 1.0)
    
    # Aplica a fórmula simultaneamente apenas nos índices onde s != 1
    a[mask] = 2.0 * (1.0 + r[mask]) / (1.0 - s[mask]) - 1.0
    
    # Onde s == 1 (o inverso da máscara), 'a' recebe -1
    a[~mask] = -1.0
    
    # 'b' é simplesmente uma cópia de 's'
    b = np.copy(s)
    
    return a, b

# Dados de teste (incluindo o caso onde s = 1)
r_teste = [-1.0, 0.0, 1.0, 0.5]
s_teste = [0.0, 0.5, 1.0, -1.0]

a_res, b_res = rstoab(r_teste, s_teste)

print("Vetor r original:", r_teste)
print("Vetor s original:", s_teste)
print("-" * 30)
print("Resultado a:", a_res)
print("Resultado b:", b_res)