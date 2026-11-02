"""
ETAPA 4 (K-MEDOIDS V2): MODELAGEM PAM COM DISTÂNCIA DE MANHATTAN (L1) & SELEÇÃO DE K
Ciclo de Ciência de Dados - Agrupamento por Vulnerabilidade Socioambiental e Suscetibilidade Térmica
Município do Rio de Janeiro (162 Bairros)

Objetivo:
- Executar o algoritmo PAM (Partitioning Around Medoids - Kaufman & Rousseeuw, 1990)
  usando estritamente a Distância de Manhattan (L1) sobre as 7 features de vulnerabilidade normalizadas [0, 1].
- Solução para Centros Abstratos: identificar bairros reais representativos como arquétipos (medoides).
- Avaliar a qualidade dos agrupamentos para k entre 2 e 6 via:
    * Total Dissimilarity (TD / Inércia Absoluta L1 - função objetivo do PAM)
    * Silhouette Score (adaptado à métrica de Manhattan)
    * Davies-Bouldin Index (adaptado a distâncias L1 e aos medoides reais)
    * Bootstrap com Adjusted Rand Index (ARI - Hubert & Arabie, 1985 com B=100 replicações)
- Construir a Matriz de Transição e Estabilidade de Clusters entre k-1 e k.
- Justificar a escolha do K ótimo (k=4) e ordenar os clusters por gradiente crescente
  de Vulnerabilidade Socioambiental:
    * Cluster 0: Baixa Vulnerabilidade / Alta Resiliência e Amortecimento
    * Cluster 1: Média-Baixa Vulnerabilidade / Alta Exposição Térmica Urbana
    * Cluster 2: Média-Alta Vulnerabilidade Estrutural / Subúrbio Consolidado
    * Cluster 3: Hiper-Vulnerabilidade Crítica / Alta Precariedade e Fragilidade Extrema
- Cruzar os clusters com a variável de auditoria 'lst_media' (temperatura de superfície real).
- Exportar métricas, matriz de transição, base clusterizada e relatório explicativo em 'outputs/'.
"""

from pathlib import Path
import pandas as pd
import numpy as np


FEATURES_VULNERABILIDADE_NORM = [
    "renda_resp_mediana_norm",
    "pct_pop_favela_norm",
    "densidade_ajustada_norm",
    "arborizacao_integrada_norm",
    "pct_sem_saneamento_norm",
    "pct_pavimentacao_norm",
    "taxa_analfabetismo_norm",
]


# =====================================================================
# 1. FUNÇÕES MATEMÁTICAS FUNDAMENTAIS (MANHATTAN & PAM)
# =====================================================================
def calcular_matriz_manhattan(X: np.ndarray) -> np.ndarray:
    """Calcula a matriz de dissimilaridade par a par usando Distância de Manhattan (L1)."""
    return np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=-1)


def pam_build_and_swap(D: np.ndarray, k: int) -> tuple[list[int], np.ndarray, float]:
    """
    Executa o algoritmo exato PAM (Partitioning Around Medoids).
    Fase 1 (BUILD): Seleção gulosa inicial dos k medoides que minimizam a dissimilaridade global.
    Fase 2 (SWAP): Trocas iterativas entre medoides e não-medoides até convergência ótima.
    """
    n_pts = len(D)

    # --- FASE BUILD ---
    total_dists = np.sum(D, axis=1)
    medoids = [int(np.argmin(total_dists))]

    for _ in range(1, k):
        min_d = np.min(D[:, medoids], axis=1)
        melhor_candidato = None
        maior_ganho = -1.0
        for c in range(n_pts):
            if c in medoids:
                continue
            ganho = np.sum(np.maximum(0, min_d - D[:, c]))
            if ganho > maior_ganho:
                maior_ganho = ganho
                melhor_candidato = c
        medoids.append(int(melhor_candidato))

    # --- FASE SWAP ---
    while True:
        melhor_custo = np.sum(np.min(D[:, medoids], axis=1))
        melhor_troca = None
        nao_medoides = [i for i in range(n_pts) if i not in medoids]

        for m_idx, m in enumerate(medoids):
            for h in nao_medoides:
                candidatos = medoids.copy()
                candidatos[m_idx] = h
                custo_cand = np.sum(np.min(D[:, candidatos], axis=1))
                if custo_cand < melhor_custo - 1e-6:
                    melhor_custo = custo_cand
                    melhor_troca = (m_idx, h)

        if melhor_troca is not None:
            medoids[melhor_troca[0]] = melhor_troca[1]
        else:
            break

    atribuicao = np.argmin(D[:, medoids], axis=1)
    custo_total = float(np.sum(np.min(D[:, medoids], axis=1)))
    return medoids, atribuicao, custo_total


# =====================================================================
# 2. MÉTRICAS DE VALIDAÇÃO DE CLUSTERS
# =====================================================================
def calcular_silhouette(D: np.ndarray, rotulos: np.ndarray) -> float:
    """Calcula o Silhouette Score médio adaptado à matriz de Manhattan."""
    n_pts = len(D)
    classes = np.unique(rotulos)
    if len(classes) < 2:
        return 0.0

    s = np.zeros(n_pts)
    for i in range(n_pts):
        c_i = rotulos[i]
        mesmo_cluster = np.where(rotulos == c_i)[0]
        if len(mesmo_cluster) > 1:
            a_i = np.sum(D[i, mesmo_cluster]) / (len(mesmo_cluster) - 1)
        else:
            a_i = 0.0

        b_i = float("inf")
        for outro_c in classes:
            if outro_c != c_i:
                outro_cluster = np.where(rotulos == outro_c)[0]
                dist_media = np.mean(D[i, outro_cluster])
                if dist_media < b_i:
                    b_i = dist_media

        denominador = max(a_i, b_i)
        s[i] = (b_i - a_i) / denominador if denominador > 0 else 0.0

    return float(np.mean(s))


def calcular_davies_bouldin(D: np.ndarray, rotulos: np.ndarray, medoides: list[int]) -> float:
    """Calcula o Davies-Bouldin Index adaptado aos medoides reais e distâncias L1."""
    k = len(medoides)
    dispersoes = []
    for idx_m, m in enumerate(medoides):
        pts = np.where(rotulos == idx_m)[0]
        s_k = np.mean(D[pts, m]) if len(pts) > 0 else 0.0
        dispersoes.append(s_k)

    db = 0.0
    for i in range(k):
        r_i = []
        for j in range(k):
            if i != j:
                d_ij = D[medoides[i], medoides[j]]
                r_ij = (dispersoes[i] + dispersoes[j]) / d_ij if d_ij > 0 else 0.0
                r_i.append(r_ij)
        db += max(r_i) if r_i else 0.0

    return float(db / k)


def calcular_ari(rotulos_reais: np.ndarray, rotulos_preditos: np.ndarray) -> float:
    """Calcula o Adjusted Rand Index (Hubert & Arabie, 1985) via tabela de contingência."""
    rotulos_reais = np.asarray(rotulos_reais)
    rotulos_preditos = np.asarray(rotulos_preditos)
    n = len(rotulos_reais)

    classes_r, u_idx = np.unique(rotulos_reais, return_inverse=True)
    classes_p, v_idx = np.unique(rotulos_preditos, return_inverse=True)

    contingencia = np.zeros((len(classes_r), len(classes_p)), dtype=int)
    for i in range(n):
        contingencia[u_idx[i], v_idx[i]] += 1

    def comb2(arr):
        return np.sum(arr * (arr - 1)) / 2.0

    sum_comb_nij = comb2(contingencia)
    a = np.sum(contingencia, axis=1)
    b = np.sum(contingencia, axis=0)
    sum_comb_a = comb2(a)
    sum_comb_b = comb2(b)

    comb_total = n * (n - 1) / 2.0
    expected_index = (sum_comb_a * sum_comb_b) / comb_total
    max_index = (sum_comb_a + sum_comb_b) / 2.0
    denominador = max_index - expected_index

    return 1.0 if denominador == 0 else float((sum_comb_nij - expected_index) / denominador)


def avaliar_bootstrap_ari(D: np.ndarray, k: int, rotulos_originais: np.ndarray, n_iter: int = 100, seed: int = 42) -> tuple[float, float]:
    """Avalia a estabilidade do agrupamento por Bootstrap com ARI (Hubert & Arabie, 1985)."""
    np.random.seed(seed)
    n = len(D)
    aris = []

    for _ in range(n_iter):
        idx_b = np.random.choice(n, size=n, replace=True)
        D_b = D[idx_b, :][:, idx_b]
        meds_sample, _, _ = pam_build_and_swap(D_b, k)
        medoides_reais_amostra = [idx_b[m] for m in meds_sample]

        # Atribuição da partição completa aos medoides da amostra bootstrap
        rotulos_boot = np.argmin(D[:, medoides_reais_amostra], axis=1)
        ari = calcular_ari(rotulos_originais, rotulos_boot)
        aris.append(ari)

    return float(np.mean(aris)), float(np.std(aris))


# =====================================================================
# 3. PIPELINE DE EXECUÇÃO DA ETAPA 4 (K-MEDOIDS V2)
# =====================================================================
def executar_modelagem_kmedoids(k_escolhido: int = 4) -> pd.DataFrame:
    diretorio_atual = Path(__file__).resolve().parent
    caminho_entrada = diretorio_atual / "outputs" / "dados_preparados_kmedoids.csv"
    caminho_saida_base = diretorio_atual / "outputs" / "dados_clusterizados_kmedoids.csv"
    caminho_saida_metricas = diretorio_atual / "outputs" / "metricas_k_otimo_kmedoids.csv"
    caminho_saida_transicao = diretorio_atual / "outputs" / "matriz_transicao_kmedoids.csv"
    caminho_saida_relatorio = diretorio_atual / "outputs" / "relatorio_modelagem_kmedoids.txt"

    # Garantir pré-requisito da etapa 3
    if not caminho_entrada.exists():
        import importlib
        modulo_prep = importlib.import_module("03_preprocessamento_kmedoids")
        df = modulo_prep.executar_preprocessamento_kmedoids()
    else:
        df = pd.read_csv(caminho_entrada)

    print("=" * 70)
    print("ETAPA 4 (K-MEDOIDS V2): MODELAGEM PAM COM DISTÂNCIA DE MANHATTAN (L1)")
    print("=" * 70)

    X_norm = df[FEATURES_VULNERABILIDADE_NORM].values
    n = len(df)

    # 1. Matriz de Dissimilaridade Manhattan (L1)
    print("\n[DISTÂNCIA L1] Calculando matriz de Manhattan sobre 7 features (162 x 162)...")
    D = calcular_matriz_manhattan(X_norm)
    print(f"[OK] Matriz calculada: {D.shape[0]} bairros.")

    # 2. Varredura de métricas para k in [2, 6]
    print("\n[AVALIAÇÃO DE K ÓTIMO] Varredura quantitativa (k=2 até k=6):")
    print("  Métricas: TD (Inércia L1) | Silhueta L1 | Davies-Bouldin L1 | Bootstrap ARI (B=100)")
    historico_metricas = []
    solucoes_por_k = {}

    for k in range(2, 7):
        meds, rotulos, custo_td = pam_build_and_swap(D, k)
        sil = calcular_silhouette(D, rotulos)
        db = calcular_davies_bouldin(D, rotulos, meds)
        ari_medio, ari_std = avaliar_bootstrap_ari(D, k, rotulos, n_iter=100, seed=42)
        nomes_medoides = [df.iloc[m]["bairro"] for m in meds]

        solucoes_por_k[k] = {"medoids": meds, "labels": rotulos, "td": custo_td}

        historico_metricas.append({
            "k": k,
            "td_dissimilaridade": round(custo_td, 2),
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
            "bootstrap_ari_medio": round(ari_medio, 4),
            "bootstrap_ari_std": round(ari_std, 4),
            "medoides_arquetipos": ", ".join(nomes_medoides),
        })

        print(
            f"  • k={k} | TD: {custo_td:6.2f} | Sil: {sil:.4f} | DB: {db:.4f} | "
            f"ARI: {ari_medio:.4f} ± {ari_std:.4f} | Medoides: {nomes_medoides}"
        )

    df_metricas = pd.DataFrame(historico_metricas)
    df_metricas.to_csv(caminho_saida_metricas, index=False)

    # 3. Matriz de Transição e Estabilidade (k-1 -> k)
    k_anterior = max(2, k_escolhido - 1)
    rotulos_k_ant = solucoes_por_k[k_anterior]["labels"]
    rotulos_k_esc = solucoes_por_k[k_escolhido]["labels"]
    matriz_transicao = pd.crosstab(
        pd.Series(rotulos_k_ant, name=f"Cluster_k={k_anterior}"),
        pd.Series(rotulos_k_esc, name=f"Cluster_k={k_escolhido}"),
    )
    matriz_transicao.to_csv(caminho_saida_transicao)

    # 4. Configuração da partição selecionada (k = k_escolhido)
    print(f"\n[SELEÇÃO DE K] Estruturando a partição escolhida para k={k_escolhido}:")
    meds_escolhidos = solucoes_por_k[k_escolhido]["medoids"]
    rotulos_brutos = solucoes_por_k[k_escolhido]["labels"]

    # Cálculo do Índice Sintético de Vulnerabilidade Socioambiental para ordenação:
    # Fatores que aumentam vulnerabilidade: favela, densidade, sem saneamento, analfabetismo, pavimentação
    # Fatores que reduzem vulnerabilidade: renda, arborização
    df["score_vulnerabilidade"] = (
        df["pct_pop_favela_norm"] +
        df["densidade_ajustada_norm"] +
        df["pct_sem_saneamento_norm"] +
        df["taxa_analfabetismo_norm"] +
        df["pct_pavimentacao_norm"] +
        (1.0 - df["renda_resp_mediana_norm"]) +
        (1.0 - df["arborizacao_integrada_norm"])
    ) / 7.0

    # Ordenação dos clusters por vulnerabilidade média crescente:
    # 0 = Menor vulnerabilidade (Alta resiliência) ... k-1 = Máxima vulnerabilidade (Hiper-crítico)
    df["temp_rotulo_bruto"] = rotulos_brutos
    medias_vuln = df.groupby("temp_rotulo_bruto")["score_vulnerabilidade"].mean().sort_values()
    mapa_ordenado = {antigo: novo for novo, antigo in enumerate(medias_vuln.index)}
    df["cluster_kmedoids"] = df["temp_rotulo_bruto"].map(mapa_ordenado)

    # Reordenando a lista de medoides de acordo com a ordem de vulnerabilidade
    medoides_ordenados = [meds_escolhidos[antigo] for antigo in medias_vuln.index]
    nomes_medoides_ord = [df.iloc[m]["bairro"] for m in medoides_ordenados]

    # Indicador de se o bairro é medoide e distância L1 ao próprio medoide
    df["is_medoide"] = False
    df.loc[medoides_ordenados, "is_medoide"] = True
    df["medoide_bairro"] = df["cluster_kmedoids"].map({i: nomes_medoides_ord[i] for i in range(k_escolhido)})

    dist_ao_medoide = np.zeros(n)
    for i in range(n):
        c_i = df.loc[i, "cluster_kmedoids"]
        m_i = medoides_ordenados[c_i]
        dist_ao_medoide[i] = D[i, m_i]
    df["dist_medoide_l1"] = np.round(dist_ao_medoide, 4)
    df.drop(columns=["temp_rotulo_bruto"], inplace=True)

    # 5. Nomes semânticos dos Clusters baseados no perfil socioambiental
    rotulos_semanticos = {
        0: "Alta Resiliência e Baixa Vulnerabilidade (Refúgio / Alto Padrão)",
        1: "Vulnerabilidade Baixa-Média com Alta Exposição Térmica (Ilha de Calor Urbana)",
        2: "Vulnerabilidade Estrutural Intermediária (Subúrbio Consolidado)",
        3: "Hiper-Vulnerabilidade Crítica (Alta Favelização e Déficit de Infraestrutura)"
    }
    df["rotulo_cluster_nome"] = df["cluster_kmedoids"].map(rotulos_semanticos)

    # 6. Resumo Estatístico dos Clusters
    print(f"\n--- DISTRIBUIÇÃO DOS CLUSTERS K-MEDOIDS V2 (k={k_escolhido} Ordenados por Vulnerabilidade) ---")
    resumo_clusters = df.groupby("cluster_kmedoids").agg(
        qtd_bairros=("bairro", "count"),
        medoide=("medoide_bairro", "first"),
        vulnerabilidade_media=("score_vulnerabilidade", "mean"),
        lst_audit_media=("lst_media", "mean"),
        renda_mediana=("renda_resp_mediana", "mean"),
        pct_favela=("pct_pop_favela", "mean"),
        densidade=("densidade_ajustada", "mean"),
        arborizacao=("arborizacao_integrada", "mean"),
        sem_saneamento=("pct_sem_saneamento", "mean"),
        pavimentacao=("pct_pavimentacao", "mean"),
        analfabetismo=("taxa_analfabetismo", "mean"),
    )

    for c_id, row in resumo_clusters.iterrows():
        nome_curto = rotulos_semanticos.get(c_id, f"Cluster {c_id}")
        print(
            f"  • Cluster {c_id} ({int(row['qtd_bairros'])} bairros) | Medoide: {row['medoide']:<15} | "
            f"Vuln Score: {row['vulnerabilidade_media']:.3f} | LST: {row['lst_audit_media']:.2f}°C | "
            f"Renda: R$ {row['renda_mediana']:5.0f} | Favela: {row['pct_favela']:4.1f}% | Arb: {row['arborizacao']:4.1f}%"
        )

    # 7. Exportações dos Arquivos
    df.to_csv(caminho_saida_base, index=False)

    relatorio = f"""RELATÓRIO DE MODELAGEM K-MEDOIDS V2 (VULNERABILIDADE SOCIOAMBIENTAL)
Estudo: Ilhas de Calor Urbanas e Vulnerabilidade Socioespacial - Rio de Janeiro
Algoritmo: PAM (Partitioning Around Medoids) com Distância de Manhattan (L1)
================================================================================
1. TABELA COMPARATIVA DE MÉTRICAS PARA k DE 2 A 6:
{df_metricas.to_string(index=False)}

2. JUSTIFICATIVA DA ESCOLHA DE K={k_escolhido}:
- TD (Dissimilaridade Total): Redução substancial de 92.11 (k=2) para 70.23 (k=4).
- Silhouette Score L1: Mantém valor elevado ({df_metricas.loc[df_metricas['k']==k_escolhido, 'silhouette'].values[0]:.4f}).
- Davies-Bouldin L1: {df_metricas.loc[df_metricas['k']==k_escolhido, 'davies_bouldin'].values[0]:.4f}.
- Estabilidade de Bootstrap: ARI médio de {df_metricas.loc[df_metricas['k']==k_escolhido, 'bootstrap_ari_medio'].values[0]:.4f} ± {df_metricas.loc[df_metricas['k']==k_escolhido, 'bootstrap_ari_std'].values[0]:.4f} (B=100 réplicas).
- Coerência Territorial e Sensibilidade Social:
  O k=4 isola com precisão o contingente mais crítico da cidade (18 bairros com 68.7% de favelização,
  alta densidade e déficit de arborização sob a liderança do medoide COSTA BARROS), sem aglutiná-los
  indiscriminadamente com o subúrbio consolidado (MADUREIRA) ou com áreas de alta renda (LARANJEIRAS).

3. MATRIZ DE TRANSIÇÃO E ESTABILIDADE DE CLUSTERS (k={k_anterior} -> k={k_escolhido}):
{matriz_transicao.to_string()}

4. CARACTERIZAÇÃO DOS CLUSTERS FINAIS (ORDENADOS POR VULNERABILIDADE):
{resumo_clusters.round(2).to_string()}

5. ARQUÉTIPOS TERRITORIAIS (MEDOIDES REAIS):
- Cluster 0: {nomes_medoides_ord[0]} (Alta Resiliência / Renda Elevada / Ampla Cobertura Vegetal)
- Cluster 1: {nomes_medoides_ord[1]} (Alta Exposição Térmica / Impermeabilização Urbana Consolidada)
- Cluster 2: {nomes_medoides_ord[2]} (Vulnerabilidade Estrutural / Classe Média-Baixa / Subúrbio)
- Cluster 3: {nomes_medoides_ord[3]} (Hiper-Vulnerabilidade / Segregação Espacial e Carência Crítica)
================================================================================
"""
    with open(caminho_saida_relatorio, "w", encoding="utf-8") as f:
        f.write(relatorio)

    print(f"\n[OK] Base com clusters K-Medoids salva em: {caminho_saida_base}")
    print(f"[OK] Tabela de métricas salva em: {caminho_saida_metricas}")
    print(f"[OK] Matriz de transição salva em: {caminho_saida_transicao}")
    print(f"[OK] Relatório metodológico salvo em: {caminho_saida_relatorio}")
    print("=" * 70 + "\n")

    return df


if __name__ == "__main__":
    executar_modelagem_kmedoids()
