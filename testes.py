import main
import numpy as np
import matplotlib.pyplot as plt
import aux_func as aux

r_teste = [-1.0, 0.0, 1.0, 0.5]
s_teste = [0.0, 0.5, 1.0, -1.0]

a_res, b_res = main.rstoab(r_teste, s_teste)

print("Vetor r original:", r_teste)
print("Vetor s original:", s_teste)
print("-" * 30)
print("Resultado a:", a_res)
print("Resultado b:", b_res)


# Se as suas funções estiverem em outro arquivo (ex: aux.py), importe-as. 
# Caso contrário, certifique-se de que estão definidas acima deste bloco.

def testar_warpfactor():
    N = 5 # Ordem do polinômio para o teste
    
    # ---------------------------------------------------------
    # TESTE 1: Verificação Numérica Extrema (Casos Críticos)
    # ---------------------------------------------------------
    pontos_criticos = np.array([-1.0, 0.0, 1.0])
    warp_critico = main.Warpfactor(N, pontos_criticos)
    
    print("--- Teste Numérico ---")
    print(f"Warp em r = -1.0 : {warp_critico[0]:.15f} (Esperado: 0.0)")
    print(f"Warp em r =  1.0 : {warp_critico[2]:.15f} (Esperado: 0.0)")
    
    # Validação automática
    assert np.isclose(warp_critico[0], 0.0), "Erro: O warp em -1 não é zero!"
    assert np.isclose(warp_critico[2], 0.0), "Erro: O warp em 1 não é zero!"
    print("Sucesso: Os limites do domínio estão ancorados corretamente.\n")

    # ---------------------------------------------------------
    # TESTE 2: Verificação Visual (Curva de Distorção)
    # ---------------------------------------------------------
    # Cria 100 pontos de alta resolução para plotar uma curva suave
    r_alta_resolucao = np.linspace(-1, 1, 100)
    warp_curva = main.Warpfactor(N, r_alta_resolucao)
    
    # Plota o resultado
    plt.figure(figsize=(8, 5))
    plt.plot(r_alta_resolucao, warp_curva, 'b-', linewidth=2, label=f'Warpfactor (N={N})')
    
    # Marcação visual do zero
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
    # Marca os pontos exatos de Gauss-Lobatto no eixo para referência
    LGL_pontos = aux.JacobiGL(0, 0, N) # Lembre-se de ajustar o 'aux.' se necessário
    plt.plot(LGL_pontos, np.zeros_like(LGL_pontos), 'ro', label='Nós de Gauss-Lobatto')
    
    plt.title("Comportamento da Função de Distorção (Warpfactor)")
    plt.xlabel("Coordenada de Referência (r)")
    plt.ylabel("Deslocamento de Warp")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# Executa o teste
testar_warpfactor()