# Análise Estrutural e Algorítmica de Redes Reais: LastFM Asia (Grafo G22)

Trabalho Prático da disciplina de **Teoria dos Grafos** — Departamento de Ciência da Computação (DCC), Universidade Federal da Bahia (UFBA).

## 1. Identificação e Seleção do Grafo
* **Matrícula:** `223118822`
* **Cálculo de Mapeamento:** $((2+2+3+1+1+8+8+2+2) \times (22 + 1)) \bmod 129 = (29 \times 23) \bmod 129 = 667 \bmod 129 = 22$
* **Dataset Atribuído:** **G22** (`feather-lastfm-social` — Rede social de usuários do LastFM da Ásia, repositório SNAP/Stanford)

## 2. Estrutura do Repositório
* `data/raw/`: Arquivos originais extraídos do SNAP (`lastfm_asia_edges.csv`, `lastfm_asia_target.csv`, `lastfm_asia_features.json`).
* `data/processed/`: Lista de arestas tratada utilizada nos experimentos (`lastfm_asia_edges_treated.csv`).
* `results/figures/`: Gráficos gerados em alta resolução (distribuição de graus, visualização da rede, escala log-log e boxplots de robustez).
* `results/tables/`: Tabelas consolidadas em CSV com os resultados das Partes I, II e III.
* `src/`: Scripts em Python (`parte1_analise_estrutural.py`, `parte2_algoritmos.py` e `parte3_analise_investigativa.py`).

## 3. Como Executar o Projeto

### Passo 1: Criar e ativar o ambiente virtual
Abra o terminal na pasta raiz do projeto e crie um ambiente virtual:
`python3 -m venv venv`

Ative o ambiente:
`source venv/bin/activate`

### Passo 2: Instalar as dependências
Com o ambiente ativo, instale as bibliotecas necessárias:
`pip install networkx igraph pandas numpy matplotlib scipy`

### Passo 3: Rodar os scripts
Execute os três arquivos em ordem:
1. `python3 src/parte1_analise_estrutural.py` (gera a base limpa, as métricas da Parte I e os dois primeiros gráficos)
2. `python3 src/parte2_algoritmos.py` (executa os 8 algoritmos 30 vezes cada e calcula os intervalos de confiança)
3. `python3 src/parte3_analise_investigativa.py` (calcula small-world, lei de potência, robustez e homofilia dos dados)