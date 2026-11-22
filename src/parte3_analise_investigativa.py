import json
import random
from collections import deque, Counter
import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
from scipy import stats

print("=== CARREGANDO DADOS PARA A PARTE III ===")
df_edges = pd.read_csv("data/processed/lastfm_asia_edges_treated.csv")
df_target = pd.read_csv("data/raw/lastfm_asia_target.csv")
with open("data/raw/lastfm_asia_features.json", "r") as f:
    features_dict = json.load(f)

G_nx = nx.from_pandas_edgelist(df_edges, source="node_1", target="node_2", create_using=nx.Graph)
n_vertices = G_nx.number_of_nodes()
n_arestas = G_nx.number_of_edges()

# ==============================================================================
# 1. SMALL-WORLD (IMPLEMENTAÇÃO PRÓPRIA SEM FUNÇÕES PRONTAS DE SMALL-WORLD)
# ==============================================================================
print("\n=== 1. VERIFICANDO PROPRIEDADE SMALL-WORLD (CÓDIGO PRÓPRIO) ===")

def construir_lista_adjacencia(arestas, num_nos):
    adj = {i: set() for i in range(num_nos)}
    for u, v in arestas:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def calcular_clusterizacao_propria(adj):
    # Calcula o coeficiente de clusterização local médio C(G) manualmente
    soma_c = 0.0
    n = len(adj)
    for v, vizinhos in adj.items():
        k = len(vizinhos)
        if k < 2:
            continue
        # Contar arestas entre os vizinhos de v
        viz_list = list(vizinhos)
        arestas_entre_vizinhos = 0
        for i in range(k):
            viz_u = adj[viz_list[i]]
            for j in range(i + 1, k):
                if viz_list[j] in viz_u:
                    arestas_entre_vizinhos += 1
        soma_c += (2.0 * arestas_entre_vizinhos) / (k * (k - 1))
    return soma_c / n

def calcular_caminho_medio_proprio(adj, fontes):
    # Calcula comprimento médio dos caminhos L(G) na maior componente via BFS próprio
    soma_distancias = 0
    total_pares = 0
    for origem in fontes:
        dist = {origem: 0}
        fila = deque([origem])
        while fila:
            u = fila.popleft()
            d_atual = dist[u]
            for w in adj[u]:
                if w not in dist:
                    dist[w] = d_atual + 1
                    fila.append(w)
        # Somar distâncias alcançadas a partir da origem
        for destino, d in dist.items():
            if destino != origem:
                soma_distancias += d
                total_pares += 1
    return soma_distancias / total_pares

adj_real = construir_lista_adjacencia(df_edges.values, n_vertices)
C_real = calcular_clusterizacao_propria(adj_real)
# Cálculo exato usando todos os 7.624 vértices como origem
L_real = calcular_caminho_medio_proprio(adj_real, list(range(n_vertices)))

# Gerar grafo aleatório equivalente G(n, m) com mesmos n e m
G_rand_nx = nx.gnm_random_graph(n_vertices, n_arestas, seed=42)
adj_rand = construir_lista_adjacencia(G_rand_nx.edges(), n_vertices)
C_rand = calcular_clusterizacao_propria(adj_rand)
# Amostra de 1.500 fontes no grafo aleatório para estimativa de alta precisão do caminho médio
random.seed(42)
fontes_rand = random.sample(range(n_vertices), 1500)
L_rand = calcular_caminho_medio_proprio(adj_rand, fontes_rand)

razao_C = C_real / C_rand
razao_L = L_real / L_rand
coef_sigma = razao_C / razao_L

print(f"Grafo Real (G22)      -> L(G) = {L_real:.4f} | C(G) = {C_real:.4f}")
print(f"Grafo Aleatório G(n,m)-> L_rand = {L_rand:.4f} | C_rand = {C_rand:.6f}")
print(f"Razões: C(G)/C_rand = {razao_C:.2f}x | L(G)/L_rand = {razao_L:.4f}x | Sigma = {coef_sigma:.2f}")

# ==============================================================================
# 2. LEI DE POTÊNCIA (POWER LAW - APÊNDICE B)
# ==============================================================================
print("\n=== 2. ANALISANDO LEI DE POTÊNCIA (POWER LAW) ===")
graus = np.array([len(adj_real[v]) for v in range(n_vertices)])
contagem_graus = Counter(graus)
ks = np.array(sorted(contagem_graus.keys()))
freqs = np.array([contagem_graus[k] for k in ks])
pk = freqs / n_vertices

# CCDF: P(K >= k)
ccdf = np.array([np.sum(graus >= k) / n_vertices for k in ks])

# Regressão linear na cauda em escala log-log (k >= 5 para evitar ruído de graus muito baixos)
mask_cauda = ks >= 5
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(ks[mask_cauda]), np.log10(pk[mask_cauda]))
gamma_estimado = -slope
r2_powerlaw = r_value ** 2

print(f"Expoente estimado da Lei de Potência (gamma): {gamma_estimado:.4f} (R² = {r2_powerlaw:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# Subplot 1: log(P(k)) vs log(k)
axes[0].scatter(ks, pk, color="#1f77b4", alpha=0.75, s=22, label="Dados empíricos P(k)")
k_fit = ks[mask_cauda]
pk_fit = (10 ** intercept) * (k_fit ** slope)
axes[0].plot(k_fit, pk_fit, color="#d62728", linestyle="--", linewidth=2,
             label=f"Ajuste Lei de Potência (γ = {gamma_estimado:.2f}, R² = {r2_powerlaw:.2f})")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_title("Distribuição de Graus P(k) em Escala Log-Log")
axes[0].set_xlabel("Grau k (escala log)")
axes[0].set_ylabel("Probabilidade P(k) (escala log)")
axes[0].grid(True, which="both", linestyle=":", alpha=0.5)
axes[0].legend()

# Subplot 2: CCDF P(K >= k) em escala log-log
axes[1].plot(ks, ccdf, color="#2ca02c", linewidth=2, label="CCDF P(K ≥ k)")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_title("Distribuição Acumulada Complementar (CCDF)")
axes[1].set_xlabel("Grau k (escala log)")
axes[1].set_ylabel("P(K ≥ k) (escala log)")
axes[1].grid(True, which="both", linestyle=":", alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.savefig("results/figures/parte3_power_law.png", dpi=300)
plt.close()

# ==============================================================================
# 3. ROBUSTEZ: REMOÇÃO ALEATÓRIA VS ATAQUE DIRECIONADO (APÊNDICE C)
# ==============================================================================
print("\n=== 3. SIMULANDO ROBUSTEZ (5% ALEATÓRIO VS 5% CENTRAIS) ===")
G_ig_base = ig.Graph.from_networkx(G_nx)
r_remover = int(0.05 * n_vertices)  # 381 vértices
n_restante = n_vertices - r_remover
print(f"Total de vértices n = {n_vertices} | Vértices removidos r (5%) = {r_remover} | Restantes = {n_restante}")

def avaliar_grafo_residual(g_ig):
    comps = g_ig.connected_components(mode="weak")
    num_comps = len(comps)
    tamanhos = comps.sizes()
    maior_comp_abs = max(tamanhos)
    frac_maior_comp = maior_comp_abs / g_ig.vcount()
    nos_isolados = sum(1 for d in g_ig.degree() if d == 0)
    frac_isolados = nos_isolados / g_ig.vcount()

    # Distâncias na maior componente residual
    sub_lcc = comps.giant()
    dist_media = sub_lcc.average_path_length(directed=False)
    diametro = sub_lcc.diameter(directed=False)
    return {
        "maior_comp_abs": maior_comp_abs,
        "S_frac_lcc": frac_maior_comp,
        "num_comps": num_comps,
        "dist_media": dist_media,
        "diametro": diametro,
        "nos_isolados": nos_isolados,
        "frac_isolados": frac_isolados
    }

# A) Remoção Aleatória (T = 30 repetições)
T_REPS = 30
random.seed(42)
metricas_rand = []
for _ in range(T_REPS):
    g_temp = G_ig_base.copy()
    remover_ids = random.sample(range(n_vertices), r_remover)
    g_temp.delete_vertices(remover_ids)
    metricas_rand.append(avaliar_grafo_residual(g_temp))

df_rand = pd.DataFrame(metricas_rand)

# B1) Ataque Direcionado por Grau (Degree Centrality)
graus_ig = G_ig_base.degree()
top_r_grau = sorted(range(n_vertices), key=lambda i: graus_ig[i], reverse=True)[:r_remover]
g_ataque_grau = G_ig_base.copy()
g_ataque_grau.delete_vertices(top_r_grau)
res_ataque_grau = avaliar_grafo_residual(g_ataque_grau)

# B2) Ataque Direcionado por Intermediação (Betweenness Centrality)
btw_ig = G_ig_base.betweenness(directed=False)
top_r_btw = sorted(range(n_vertices), key=lambda i: btw_ig[i], reverse=True)[:r_remover]
g_ataque_btw = G_ig_base.copy()
g_ataque_btw.delete_vertices(top_r_btw)
res_ataque_btw = avaliar_grafo_residual(g_ataque_btw)

# Consolidar tabela comparativa de robustez
tabela_robustez = pd.DataFrame([
    {
        "Cenário": "Grafo Original (Intacto)",
        "Maior Comp. (Vértices)": n_vertices,
        "Fração Maior Comp. (S)": 1.0000,
        "Nº Componentes": 1,
        "Caminho Médio (L)": round(L_real, 4),
        "Diâmetro (D)": 15,
        "Fração Isolados": 0.0000
    },
    {
        "Cenário": f"Remoção Aleatória 5% (Média ± DP, T={T_REPS})",
        "Maior Comp. (Vértices)": f"{df_rand['maior_comp_abs'].mean():.1f} ± {df_rand['maior_comp_abs'].std():.1f}",
        "Fração Maior Comp. (S)": f"{df_rand['S_frac_lcc'].mean():.4f} ± {df_rand['S_frac_lcc'].std():.4f}",
        "Nº Componentes": f"{df_rand['num_comps'].mean():.1f} ± {df_rand['num_comps'].std():.1f}",
        "Caminho Médio (L)": f"{df_rand['dist_media'].mean():.4f} ± {df_rand['dist_media'].std():.4f}",
        "Diâmetro (D)": f"{df_rand['diametro'].mean():.1f} ± {df_rand['diametro'].std():.1f}",
        "Fração Isolados": f"{df_rand['frac_isolados'].mean():.4f} ± {df_rand['frac_isolados'].std():.4f}"
    },
    {
        "Cenário": "Ataque Direcionado 5% (Maior Grau)",
        "Maior Comp. (Vértices)": res_ataque_grau["maior_comp_abs"],
        "Fração Maior Comp. (S)": round(res_ataque_grau["S_frac_lcc"], 4),
        "Nº Componentes": res_ataque_grau["num_comps"],
        "Caminho Médio (L)": round(res_ataque_grau["dist_media"], 4),
        "Diâmetro (D)": res_ataque_grau["diametro"],
        "Fração Isolados": round(res_ataque_grau["frac_isolados"], 4)
    },
    {
        "Cenário": "Ataque Direcionado 5% (Maior Betweenness)",
        "Maior Comp. (Vértices)": res_ataque_btw["maior_comp_abs"],
        "Fração Maior Comp. (S)": round(res_ataque_btw["S_frac_lcc"], 4),
        "Nº Componentes": res_ataque_btw["num_comps"],
        "Caminho Médio (L)": round(res_ataque_btw["dist_media"], 4),
        "Diâmetro (D)": res_ataque_btw["diametro"],
        "Fração Isolados": round(res_ataque_btw["frac_isolados"], 4)
    }
])

print(tabela_robustez.to_string(index=False))
tabela_robustez.to_csv("results/tables/parte3_robustez.csv", index=False)

# Gerar Boxplots das 4 métricas exigidas no Apêndice C comparando com os ataques direcionados
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
configs_box = [
    ("S_frac_lcc", "Métrica A: Fração da Maior Componente (S)", axes[0, 0]),
    ("num_comps", "Métrica B: Número de Componentes Conexas", axes[0, 1]),
    ("dist_media", "Métrica C: Comprimento Médio dos Caminhos (L)", axes[1, 0]),
    ("frac_isolados", "Métrica D: Fração de Nós Isolados", axes[1, 1]),
]

for col, titulo, ax in configs_box:
    ax.boxplot(df_rand[col], positions=[1], widths=0.4, patch_artist=True,
               boxprops=dict(facecolor="#a6cee3", color="#1f78b4"),
               medianprops=dict(color="black", linewidth=1.5))
    ax.scatter([2], [res_ataque_grau[col]], color="#e31a1c", s=80, zorder=5, label="Ataque (Grau)")
    ax.scatter([3], [res_ataque_btw[col]], color="#ff7f00", s=80, marker="D", zorder=5, label="Ataque (Betweenness)")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Aleatório\n(30 rep.)", "Ataque\n(Grau)", "Ataque\n(Betweenness)"])
    ax.set_title(titulo)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/figures/parte3_robustez_boxplots.png", dpi=300)
plt.close()

# ==============================================================================
# 4. DESCOBERTA INTERESSANTE (HOMOFILIA GEOGRÁFICA E GOSTO MUSICAL)
# ==============================================================================
print("\n=== 4. INVESTIGANDO DESCOBERTA NOS METADADOS (PAÍS E ARTISTAS) ===")
pais_map = dict(zip(df_target["id"], df_target["target"]))
artistas_sets = {int(k): set(v) for k, v in features_dict.items()}

# 1) Homofilia por país nas arestas conectadas vs pares aleatórios
arestas_mesmo_pais = sum(1 for u, v in df_edges.values if pais_map[u] == pais_map[v])
taxa_homofilia_real = arestas_mesmo_pais / n_arestas

# Expectativa aleatória (probabilidade de dois nós ao acaso serem do mesmo país)
contagem_paises = df_target["target"].value_counts()
prob_mesmo_pais_acaso = sum((c / n_vertices) ** 2 for c in contagem_paises)

# 2) Similaridade de Jaccard dos artistas curtidos entre amigos vs não-amigos
def jaccard(s1, s2):
    uniao = len(s1 | s2)
    return len(s1 & s2) / uniao if uniao > 0 else 0.0

jaccards_amigos = [jaccard(artistas_sets[u], artistas_sets[v]) for u, v in df_edges.values]
random.seed(42)
pares_nao_amigos = []
while len(pares_nao_amigos) < 27806:
    u = random.randint(0, n_vertices - 1)
    v = random.randint(0, n_vertices - 1)
    if u != v and v not in adj_real[u]:
        pares_nao_amigos.append((u, v))
jaccards_nao_amigos = [jaccard(artistas_sets[u], artistas_sets[v]) for u, v in pares_nao_amigos]

print(f"Arestas ligando usuários do MESMO país: {arestas_mesmo_pais}/{n_arestas} ({taxa_homofilia_real*100:.2f}%)")
print(f"Probabilidade esperada ao acaso (mesmo país): {prob_mesmo_pais_acaso*100:.2f}% (Fator: {taxa_homofilia_real/prob_mesmo_pais_acaso:.2f}x)")
print(f"Similaridade média de artistas (Jaccard) entre AMIGOS: {np.mean(jaccards_amigos):.4f}")
print(f"Similaridade média de artistas (Jaccard) entre NÃO-AMIGOS: {np.mean(jaccards_nao_amigos):.4f} (Fator: {np.mean(jaccards_amigos)/np.mean(jaccards_nao_amigos):.2f}x)")

# Salvar resumo geral da Parte III
resumo_p3 = pd.DataFrame([
    ("C(G) Real vs C_rand", f"{C_real:.4f} vs {C_rand:.6f} ({razao_C:.2f}x)"),
    ("L(G) Real vs L_rand", f"{L_real:.4f} vs {L_rand:.4f} ({razao_L:.4f}x)"),
    ("Coeficiente Small-World (Sigma)", f"{coef_sigma:.2f}"),
    ("Expoente Power Law (gamma)", f"{gamma_estimado:.4f} (R²={r2_powerlaw:.4f})"),
    ("Homofilia Geográfica (Mesmo País)", f"{taxa_homofilia_real*100:.2f}% vs {prob_mesmo_pais_acaso*100:.2f}% ao acaso"),
    ("Jaccard Artistas (Amigos vs Não-Amigos)", f"{np.mean(jaccards_amigos):.4f} vs {np.mean(jaccards_nao_amigos):.4f}")
], columns=["Indicador", "Resultado"])
resumo_p3.to_csv("results/tables/parte3_resumo_investigativo.csv", index=False)
print("\nArquivos da Parte III salvos com sucesso!")