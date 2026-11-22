import time
import math
import random
import pandas as pd
import numpy as np
import networkx as nx
from collections import deque

print("=== CARREGANDO GRAFO TRATADO ===")
df_edges = pd.read_csv("data/processed/lastfm_asia_edges_treated.csv")
G = nx.from_pandas_edgelist(df_edges, source="node_1", target="node_2", create_using=nx.Graph)

# Atribuir peso unitário (1.0) nas arestas para algoritmos ponderados (Dijkstra, Bellman-Ford, Floyd-Warshall, Kruskal)
for u, v in G.edges():
    G[u][v]["weight"] = 1.0

V_total = G.number_of_nodes()
E_total = G.number_of_edges()
print(f"Grafo completo: |V| = {V_total}, |E| = {E_total}")

# Criar subgrafo conexo de 400 vértices (via BFS a partir do vértice de maior grau) para algoritmos cúbicos
hub_principal = max(G.degree(), key=lambda x: x[1])[0]
nos_subgrafo = list(nx.bfs_tree(G, source=hub_principal).nodes())[:400]
G_sub = G.subgraph(nos_subgrafo).copy()
V_sub = G_sub.number_of_nodes()
E_sub = G_sub.number_of_edges()
print(f"Subgrafo induzido (para Bellman-Ford e Floyd-Warshall): |V_sub| = {V_sub}, |E_sub| = {E_sub}\n")

# Converter para representação orientada apenas para o algoritmo de Tarjan (SCC)
G_dir = G.to_directed()

# --- IMPLEMENTAÇÕES E WRAPPERS DOS 8 ALGORITMOS ---

def run_bfs(grafo, origem):
    visitados = {origem}
    fila = deque([origem])
    while fila:
        u = fila.popleft()
        for v in grafo.neighbors(u):
            if v not in visitados:
                visitados.add(v)
                fila.append(v)
    return len(visitados)

def run_dfs(grafo, origem):
    visitados = set()
    pilha = [origem]
    while pilha:
        u = pilha.pop()
        if u not in visitados:
            visitados.add(u)
            for v in grafo.neighbors(u):
                if v not in visitados:
                    pilha.append(v)
    return len(visitados)

def run_eulerianidade(grafo):
    # Para grafo não direcionado conexo: é euleriano sse todos os vértices têm grau par
    impares = sum(1 for _, d in grafo.degree() if d % 2 != 0)
    return (impares == 0), impares

def run_dijkstra(grafo, origem):
    distancias = nx.single_source_dijkstra_path_length(grafo, origem, weight="weight")
    return len(distancias)

def run_bellman_ford(grafo, origem):
    distancias = nx.single_source_bellman_ford_path_length(grafo, origem, weight="weight")
    return len(distancias)

def run_floyd_warshall(grafo):
    # Matriz numpy via NetworkX para Floyd-Warshall
    dist_matrix = nx.floyd_warshall_numpy(grafo, weight="weight")
    return dist_matrix.shape[0]

def run_tarjan(grafo_direcionado):
    # Componentes Fortemente Conexas via algoritmo de Tarjan
    sccs = list(nx.strongly_connected_components(grafo_direcionado))
    return len(sccs)

def run_kruskal(grafo):
    agm = nx.minimum_spanning_tree(grafo, weight="weight", algorithm="kruskal")
    return agm.number_of_edges()

# --- BENCHMARK ESTATÍSTICO (N = 30 EXECUÇÕES) ---
N_REPETICOES = 30
Z_95 = 1.96  # Valor crítico da Normal padrão para 95% de confiança (n >= 30)

random.seed(42)
fontes_completas = random.sample(list(G.nodes()), N_REPETICOES)
fontes_subgrafo = random.sample(list(G_sub.nodes()), N_REPETICOES)

experimentos = [
    ("1. Busca em Largura (BFS)", "O(|V| + |E|)", f"Completo (|V|={V_total})", lambda i: run_bfs(G, fontes_completas[i])),
    ("2. Busca em Profundidade (DFS)", "O(|V| + |E|)", f"Completo (|V|={V_total})", lambda i: run_dfs(G, fontes_completas[i])),
    ("3. Verificação de Eulerianidade", "O(|V|)", f"Completo (|V|={V_total})", lambda i: run_eulerianidade(G)),
    ("4. Dijkstra (caminho mínimo 1-N)", "O((|V| + |E|) log |V|)", f"Completo (|V|={V_total})", lambda i: run_dijkstra(G, fontes_completas[i])),
    ("5. Bellman-Ford (caminho mínimo 1-N)", "O(|V| · |E|)", f"Subgrafo (|V|={V_sub})", lambda i: run_bellman_ford(G_sub, fontes_subgrafo[i])),
    ("6. Floyd-Warshall (todos os pares)", "O(|V|³)", f"Subgrafo (|V|={V_sub})", lambda i: run_floyd_warshall(G_sub)),
    ("7. Algoritmo de Tarjan (SCC)", "O(|V| + |E|)", f"Completo Dir. (|V|={V_total})", lambda i: run_tarjan(G_dir)),
    ("8. Árvore Geradora Mínima (Kruskal)", "O(|E| log |E|)", f"Completo (|V|={V_total})", lambda i: run_kruskal(G)),
]

resultados = []

print("=== INICIANDO EXECUÇÕES (30 REPETIÇÕES POR ALGORITMO) ===")
for nome, comp_teorica, escopo, func in experimentos:
    tempos_ms = []
    ultimo_retorno = None
    for i in range(N_REPETICOES):
        t0 = time.perf_counter()
        ultimo_retorno = func(i)
        t1 = time.perf_counter()
        tempos_ms.append((t1 - t0) * 1000.0)  # converter para milissegundos

    media = float(np.mean(tempos_ms))
    desvio = float(np.std(tempos_ms, ddof=1))  # desvio padrão amostral
    margem_erro = Z_95 * (desvio / math.sqrt(N_REPETICOES))
    ic_inf = max(0.0, media - margem_erro)
    ic_sup = media + margem_erro

    print(f"{nome:38s} | Média: {media:9.4f} ms | DP: {desvio:8.4f} ms | IC 95%: [{ic_inf:9.4f}, {ic_sup:9.4f}] | Retorno: {ultimo_retorno}")

    resultados.append({
        "Algoritmo": nome,
        "Complexidade Teórica": comp_teorica,
        "Escopo": escopo,
        "Média (ms)": round(media, 4),
        "Desvio Padrão (ms)": round(desvio, 4),
        "IC 95% Inferior (ms)": round(ic_inf, 4),
        "IC 95% Superior (ms)": round(ic_sup, 4),
        "Resultado/Saída": str(ultimo_retorno)
    })

df_res = pd.DataFrame(resultados)
df_res.to_csv("results/tables/parte2_tempos_ic.csv", index=False)
print("\nTabela da Parte II salva em: results/tables/parte2_tempos_ic.csv")