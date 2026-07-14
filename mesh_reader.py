import numpy as np

def MeshReader2D(nome_arquivo):
    """
    Lê um arquivo .msh do Gmsh (Version 2 ASCII) e extrai 
    as coordenadas dos vértices (VX, VY) e a conectividade (EToV).
    """
    with open(nome_arquivo, 'r') as f:
        linhas = f.readlines()
    VX_list = []
    VY_list = []
    EToV_list = []
    BCTags_list = [] # Vai guardar as linhas das bordas
    
    i = 0
    while i < len(linhas):
        linha = linhas[i].strip()
        
        # ----------------------------------------------------
        # BLOCO 1: Lendo as Coordenadas dos Nós (VX e VY)
        # ----------------------------------------------------
        if linha == '$Nodes':
            i += 1
            num_nodes = int(linhas[i].strip())
            
            # Prepara as listas de coordenadas
            VX = np.zeros(num_nodes)
            VY = np.zeros(num_nodes)
            
            for _ in range(num_nodes):
                i += 1
                dados = linhas[i].split()
                node_id = int(dados[0]) - 1 # Subtrai 1 para o Python!
                
                VX[node_id] = float(dados[1])
                VY[node_id] = float(dados[2])
                
        # ----------------------------------------------------
        # BLOCO 2: Lendo a Conectividade (EToV)
        # ----------------------------------------------------
        elif linha == '$Elements':
            i += 1
            num_elements = int(linhas[i].strip())
            
            for _ in range(num_elements):
                i += 1
                dados = linhas[i].split()
                
                tipo_elemento = int(dados[1])
                num_tags = int(dados[2])
                
                # O Gmsh lista as tags antes dos nós. 
                # Geralmente a primeira tag é o ID do Grupo Físico que criamos no .geo
                tag_fisica = int(dados[3]) 
                
                # Pega os números dos nós pulando as tags iniciais (subtraindo 1 para o Python)
                nos = [int(n) - 1 for n in dados[3 + num_tags:]]
                
                if tipo_elemento == 2:
                    # Tipo 2 é um Triângulo! Salva na EToV
                    EToV_list.append(nos)
                    
                elif tipo_elemento == 1:
                    # Tipo 1 é uma Linha (Borda). Guardamos para montar o BCType depois
                    BCTags_list.append({"nos": nos, "tag": tag_fisica})
                    
        i += 1
        
    EToV = np.array(EToV_list)
    
    # K é o número total de triângulos
    K = EToV.shape[0]
    
    print(f"Malha carregada com sucesso!")
    print(f"Número de Nós: {len(VX)}")
    print(f"Número de Elementos (K): {K}")
    
    return VX, VY, EToV, BCTags_list