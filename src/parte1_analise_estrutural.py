import os
import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt

# Garantir que as pastas de saída existam
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

print("=== 1. CARREGAMENTO E TRATAMENTO DE DADOS ===")
df_edges = pd.read_csv("data/raw/lastfm_asia_edges.csv")
df_target = pd.read_csv("data/raw/lastfm_asia_target.csv")

print(f"Arestas brutas no CSV: {len(df_edges)}")
print(f"Vértices no arquivo target: {len(df_target)}")

# Verificar auto-loops (u == v)
auto_loops = df_edges[df_edges["node_1"] == df_edges["node_2"]]
print(f"Auto-loops encontrados: {len(auto_loops)}")

# Remover auto-loops e padronizar arestas não direcionadas (min(u,v), max(u,v)) para checar duplicatas
df_clean = df_edges[df_edges["node_1"] != df_edges["node_2"]].copy()
u_min = np.minimum(df_clean["node_1"], df_clean["node_2"])
v_max = np.maximum(df_clean["node_1"], df_clean["node_2"])
df_clean["node_1"] = u_min
df_clean["node_2"] = v_max

duplicadas = df_clean.duplicated().sum()
print(f"Arestas duplicadas (multi-arestas) encontradas: {duplicadas}")
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

# Construir grafo no NetworkX garantindo todos os 7.624 vértices (inclusive eventuais isolados)
G_nx = nx.Graph()
G_nx.add_nodes_from(df_target["id"].tolist())
G_nx.add_edges_from(df_clean[["node_1", "node_2"]].itertuples(index=False, name=None))

# Verificar componentes conexas
componentes = sorted(nx.connected_components(G_nx), key=len, reverse=True)
num_componentes = len(componentes)
tamanhos_componentes = [len(c) for c in componentes]
print(f"Número de componentes conexas: {num_componentes}")
print(f"Tamanhos das componentes: {tamanhos_componentes}")

# Extrair a maior componente conexa (LCC) caso o grafo seja desconexo
if num_componentes > 1:
    maior_comp = componentes[0]
    G_lcc = G_nx.subgraph(maior_comp).copy()
    print(f"Extraindo maior componente conexa: {G_lcc.number_of_nodes()} vértices, {G_lcc.number_of_edges()} arestas.")
else:
    G_lcc = G_nx.copy()
    print("O grafo já é totalmente conexo (1 única componente).")

# Salvar dataset tratado conforme exigido no relatório
df_processed_edges = nx.to_pandas_edgelist(G_lcc)
df_processed_edges.columns = ["node_1", "node_2"]
df_processed_edges.to_csv("data/processed/lastfm_asia_edges_treated.csv", index=False)
print("Grafo tratado salvo em: data/processed/lastfm_asia_edges_treated.csv\n")

print("=== 2. CÁLCULO DAS MÉTRICAS ESTRUTURAIS (PARTE I) ===")
# Converter para igraph para cálculo rápido de distâncias exatas (diâmetro, raio, caminho médio)
G_ig = ig.Graph.from_networkx(G_lcc)

n_vertices = G_ig.vcount()
n_arestas = G_ig.ecount()
graus = np.array(G_ig.degree())
grau_min = int(graus.min())
grau_max = int(graus.max())
grau_medio = float(graus.mean())
densidade = float(G_ig.density())

# Distâncias (executadas na maior componente conexa)
diametro = int(G_ig.diameter(directed=False))
raio = int(G_ig.radius())
caminho_medio = float(G_ig.average_path_length(directed=False))

# Clusterização e triângulos
clusterizacao_media = float(G_ig.transitivity_avglocal_undirected(mode="zero"))
clusterizacao_global = float(G_ig.transitivity_undirected())
num_triangulos = len(G_ig.cliques(min=3, max=3))

metricas = [
    ("Número de vértices (|V|)", f"{n_vertices}", "Sim"),
    ("Número de arestas (|E|)", f"{n_arestas}", "Sim"),
    ("Grau mínimo", f"{grau_min}", "Sim"),
    ("Grau máximo", f"{grau_max}", "Sim"),
    ("Grau médio", f"{grau_medio:.4f}", "Sim"),
    ("Densidade", f"{densidade:.6f}", "Sim"),
    ("Número de componentes conexas (original)", f"{num_componentes}", "Sim"),
    ("Tamanho das componentes (original)", f"{tamanhos_componentes}", "Sim"),
    ("Diâmetro", f"{diametro}", "Sim"),
    ("Raio", f"{raio}", "Sim"),
    ("Comprimento médio dos caminhos", f"{caminho_medio:.4f}", "Sim"),
    ("Coeficiente de clusterização médio (local)", f"{clusterizacao_media:.4f}", "Sim"),
    ("Coeficiente de clusterização global (transitividade)", f"{clusterizacao_global:.4f}", "Sim"),
    ("Número de triângulos", f"{num_triangulos}", "Sim"),
]

df_metricas = pd.DataFrame(metricas, columns=["Métrica", "Valor", "Computada"])
print(df_metricas.to_string(index=False))
df_metricas.to_csv("results/tables/parte1_metricas.csv", index=False)

print("\n=== 3. GERANDO GRÁFICOS DA PARTE I ===")
# Gráfico 1: Distribuição de graus (escala linear)
plt.figure(figsize=(8, 4.5))
plt.hist(graus, bins=50, color="#2b5c8f", edgecolor="black", alpha=0.85)
plt.title("Distribuição de Graus - Rede Social LastFM Asia (G22)")
plt.xlabel("Grau (k)")
plt.ylabel("Frequência (Número de vértices)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/figures/parte1_distribuicao_graus.png", dpi=300)
plt.close()

# Gráfico 2: Visualização reduzida do grafo (subgrafo induzido pelos 250 nós de maior grau para legibilidade)
top_nodes = sorted(G_lcc.degree(), key=lambda x: x[1], reverse=True)[:250]
top_node_ids = [n for n, d in top_nodes]
G_vis = G_lcc.subgraph(top_node_ids).copy()

# Mapear país (target) para colorir os vértices
target_map = dict(zip(df_target["id"], df_target["target"]))
node_colors = [target_map.get(n, 0) for n in G_vis.nodes()]
node_sizes = [15 + G_vis.degree(n) * 1.5 for n in G_vis.nodes()]

plt.figure(figsize=(9, 7))
pos = nx.spring_layout(G_vis, seed=42, k=0.25)
nx.draw_networkx_edges(G_vis, pos, alpha=0.15, edge_color="gray", width=0.6)
nodes = nx.draw_networkx_nodes(
    G_vis, pos,
    node_color=node_colors,
    cmap=plt.cm.tab20,
    node_size=node_sizes,
    alpha=0.9
)
plt.title("Visualização do Núcleo da Rede (Top 250 vértices por grau, coloridos por país)")
plt.axis("off")
plt.tight_layout()
plt.savefig("results/figures/parte1_visualizacao_grafo.png", dpi=300)
plt.close()

print("Figuras salvas em results/figures/ e tabela salva em results/tables/parte1_metricas.csv.")