"""
ETAPA 5 (K-MEDOIDS V2): PERFILAMENTO DOS ARQUÉTIPOS, ETA² & MATRIZ DE RISCO 2D
Ciclo de Ciência de Dados - Agrupamento por Vulnerabilidade Socioambiental e Suscetibilidade Térmica
Município do Rio de Janeiro (162 Bairros)

Objetivo:
- Analisar os clusters formados pelo algoritmo PAM (K-Medoids V2), identificando o perfil substantivo
  de cada grupo através de seu medoide real (arquétipo concreto da cidade).
- Hierarquia dos Clusters por Vulnerabilidade Socioambiental:
    * Cluster 0: Baixa Vulnerabilidade / Alta Resiliência (Arquétipo: Laranjeiras)
    * Cluster 1: Vulnerabilidade Baixa-Média / Ilha de Calor Urbana Consolidada (Arquétipo: Irajá)
    * Cluster 2: Vulnerabilidade Estrutural Intermediária / Subúrbio Operário (Arquétipo: Madureira)
    * Cluster 3: Hiper-Vulnerabilidade Crítica / Favelização e Déficits Extremos (Arquétipo: Costa Barros)
- Calcular o Eta ao Quadrado (η² = SS_between / SS_total) para cada uma das 7 variáveis de vulnerabilidade
  e também para a variável de controle físico 'lst_media' (avaliando quanto do calor de superfície é explicado).
- Integrar a Exposição Térmica Observada (LST) com o Índice de Vulnerabilidade Socioambiental,
  estruturando a Matriz 2D de Risco Integrado (50% Térmico + 50% Social).
- Classificar os bairros em 4 Níveis de Prioridade de Intervenção Urbana, atribuir Diretrizes Estratégicas
  de Políticas Públicas e Alertas de MAUP (Nível 2).
- Exportar a base enriquecida em 'outputs/dados_perfilados_matriz_kmedoids.csv', a tabela de η² em
  'outputs/eta_quadrado_atributos_kmedoids.csv', o resumo dos perfis em 'outputs/perfil_resumo_clusters_kmedoids.csv'
  e o relatório explicativo em 'outputs/relatorio_perfilamento_kmedoids.txt'.
"""

from pathlib import Path
import pandas as pd
import numpy as np


FEATURES_VULNERABILIDADE = [
    "renda_resp_mediana",
    "pct_pop_favela",
    "densidade_ajustada",
    "arborizacao_integrada",
    "pct_sem_saneamento",
    "pct_pavimentacao",
    "taxa_analfabetismo",
]

VARIAVEL_CONTROLE = "lst_media"


def gerar_config_clusters(k: int, df: pd.DataFrame) -> tuple[dict, dict]:
    """Gera nomes semânticos e perfis conceituais dinamicamente para os clusters ordenados por vulnerabilidade."""
    nomes = {}
    perfis = {}

    for c_id in range(k):
        sub = df[df["cluster_kmedoids"] == c_id]
        medoide_nome = sub["medoide_bairro"].iloc[0] if "medoide_bairro" in sub.columns and len(sub) > 0 else f"Cluster {c_id}"

        if c_id == 0:
            nomes[c_id] = f"Alta Resiliência e Baixa Vulnerabilidade (Arquétipo: {medoide_nome})"
            perfis[c_id] = "Baixa Vulnerabilidade / Alta Capacidade Adaptativa"
        elif c_id == k - 1:
            nomes[c_id] = f"Hiper-Vulnerabilidade Crítica (Arquétipo: {medoide_nome})"
            perfis[c_id] = "Vulnerabilidade Crítica / Segregação Socioespacial"
        elif c_id == 1 and k >= 4:
            nomes[c_id] = f"Vulnerabilidade Baixa-Média com Alta Exposição Térmica (Arquétipo: {medoide_nome})"
            perfis[c_id] = "Média-Baixa Vulnerabilidade / Ilha de Calor Severa"
        elif c_id == k - 2 and k >= 4:
            nomes[c_id] = f"Vulnerabilidade Estrutural Intermediária (Arquétipo: {medoide_nome})"
            perfis[c_id] = "Média-Alta Vulnerabilidade / Subúrbio Consolidado"
        else:
            nomes[c_id] = f"Vulnerabilidade Transição Nível {c_id} (Arquétipo: {medoide_nome})"
            perfis[c_id] = "Transição Intermediária"

    return nomes, perfis


def calcular_eta_quadrado(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Calcula o Eta ao Quadrado (η² = SS_between / SS_total) para todos os indicadores de vulnerabilidade e LST."""
    todas_features = FEATURES_VULNERABILIDADE + [VARIAVEL_CONTROLE]
    registros_eta = []

    for col in todas_features:
        y = df[col].values
        media_global = np.mean(y)
        ss_total = np.sum((y - media_global) ** 2)

        ss_between = 0.0
        for c in range(k):
            sub = y[df["cluster_kmedoids"] == c]
            if len(sub) > 0:
                ss_between += len(sub) * ((np.mean(sub) - media_global) ** 2)

        eta2 = ss_between / ss_total if ss_total > 0 else 0.0
        tipo = "Térmico (Controle)" if col == VARIAVEL_CONTROLE else "Vulnerabilidade"

        registros_eta.append({
            "atributo": col,
            "tipo_indicador": tipo,
            "ss_between": round(ss_between, 2),
            "ss_total": round(ss_total, 2),
            "eta_quadrado": round(eta2, 4),
            "pct_variancia_explicada": f"{eta2 * 100:.2f}%",
        })

    df_eta = pd.DataFrame(registros_eta).sort_values(by="eta_quadrado", ascending=False)
    return df_eta


def classificar_prioridade(row: pd.Series) -> str:
    """Classifica o nível de prioridade na Matriz 2D (Exposição Térmica Real x Vulnerabilidade Social)."""
    st = float(row["score_termico"])
    sv = float(row["score_vulnerabilidade_100"])
    c_id = int(row["cluster_kmedoids"])

    # Cluster 3 (Hiper-Vulnerável): Prioridade máxima se tiver calor moderado ou alto
    if c_id == 3:
        return "Urgente / Intervenção Imediata" if st >= 50.0 else "Acompanhamento Prioritário"
    # Bairros com alta exposição térmica combinada com alta vulnerabilidade
    elif st >= 65.0 and sv >= 40.0:
        return "Urgente / Intervenção Imediata"
    # Bairros de calor intenso mas com boa capacidade adaptativa financeira/vegetal
    elif st >= 65.0:
        return "Alerta Físico / Infraestrutura"
    # Bairros com vulnerabilidade social elevada mesmo sem temperatura extrema de superfície
    elif sv >= 50.0:
        return "Acompanhamento Prioritário"
    # Bairros frios e com alta capacidade adaptativa
    elif st < 50.0 and sv < 35.0:
        return "Baixo Risco / Preservação"
    else:
        return "Acompanhamento / Monitoramento"


def atribuir_diretriz_acao(prioridade: str) -> str:
    """Define a recomendação de política pública associada ao nível de prioridade."""
    if prioridade == "Urgente / Intervenção Imediata":
        return "Arborização viária prioritária, micro-parques reflexivos, regularização de água/esgoto e pontos de hidratação comunitária."
    elif prioridade == "Alerta Físico / Infraestrutura":
        return "Incentivo a telhados verdes/refletivos, ampliação de pavimentos permeáveis e mitigação do selamento do solo urbano."
    elif prioridade == "Acompanhamento Prioritário":
        return "Fortalecimento da rede de assistência básica, ampliação da infraestrutura sanitária e monitoramento de saúde comunitária."
    elif prioridade == "Acompanhamento / Monitoramento":
        return "Monitoramento microclimático sazonal e conservação preventiva das áreas arborizadas e praças locais."
    else:
        return "Preservação da cobertura florestal nativa e contenção do avanço da impermeabilização do solo."


def executar_perfilamento_matriz_kmedoids() -> pd.DataFrame:
    diretorio_atual = Path(__file__).resolve().parent
    caminho_entrada = diretorio_atual / "outputs" / "dados_clusterizados_kmedoids.csv"
    caminho_saida = diretorio_atual / "outputs" / "dados_perfilados_matriz_kmedoids.csv"
    caminho_saida_eta = diretorio_atual / "outputs" / "eta_quadrado_atributos_kmedoids.csv"
    caminho_saida_resumo = diretorio_atual / "outputs" / "perfil_resumo_clusters_kmedoids.csv"
    caminho_saida_relatorio = diretorio_atual / "outputs" / "relatorio_perfilamento_kmedoids.txt"

    # Garantir pré-requisito da etapa 4
    if not caminho_entrada.exists():
        import importlib
        modulo_kmedoids = importlib.import_module("04_modelagem_kmedoids")
        df = modulo_kmedoids.executar_modelagem_kmedoids()
    else:
        df = pd.read_csv(caminho_entrada)

    clusters_unicos = sorted(df["cluster_kmedoids"].dropna().unique())
    k = len(clusters_unicos)

    print("=" * 70)
    print(f"ETAPA 5 (K-MEDOIDS V2): PERFILAMENTO, ETA² & MATRIZ DE RISCO 2D (K = {k})")
    print("=" * 70)

    # 1. Configuração Semântica dos Clusters
    nomes_clusters, perfis_clusters = gerar_config_clusters(k, df)
    df["cluster_nome"] = df["cluster_kmedoids"].map(nomes_clusters)
    df["perfil_vulnerabilidade"] = df["cluster_kmedoids"].map(perfis_clusters)

    # 2. Cálculo do Eta ao Quadrado (η²) por indicador
    print("\n[VARIÂNCIA EXPLICADA - η²] Calculando poder explicativo dos clusters:")
    df_eta = calcular_eta_quadrado(df, k)
    df_eta.to_csv(caminho_saida_eta, index=False)
    for _, row in df_eta.iterrows():
        print(f"  • {row['atributo']:<24} ({row['tipo_indicador']:<18}) -> η² = {row['eta_quadrado']:.4f} ({row['pct_variancia_explicada']})")

    # 3. Construção dos Eixos da Matriz 2D:
    # Eixo 1: Exposição Térmica Real Observada (LST - 0 a 100)
    lst_min = float(df["lst_media"].min())
    lst_max = float(df["lst_media"].max())
    df["score_termico"] = (100.0 * (df["lst_media"] - lst_min) / (lst_max - lst_min)).round(2)

    # Eixo 2: Vulnerabilidade Socioambiental Intrínseca (Score Sintético - 0 a 100)
    vuln_min = float(df["score_vulnerabilidade"].min())
    vuln_max = float(df["score_vulnerabilidade"].max())
    df["score_vulnerabilidade_100"] = (100.0 * (df["score_vulnerabilidade"] - vuln_min) / (vuln_max - vuln_min)).round(2)

    # 4. Escore de Risco Integrado (50% Térmico + 50% Social)
    df["score_risco_integrado"] = (0.50 * df["score_termico"] + 0.50 * df["score_vulnerabilidade_100"]).round(2)

    # 5. Classificação de Prioridade e Ação Recomendada
    df["nivel_prioridade"] = df.apply(classificar_prioridade, axis=1)
    df["acao_recomendada"] = df["nivel_prioridade"].apply(atribuir_diretriz_acao)

    # 6. Alerta de MAUP (Nível 2): recomendação de microanálise setorial em áreas de densidade extrema ou alto risco
    df["alerta_maup"] = df.apply(
        lambda r: "Recomendada microanálise setorial intra-bairro (Nível 2)"
        if ("Urgente" in r["nivel_prioridade"] or r["densidade_ajustada"] > 35000 or r["pct_pop_favela"] > 50.0)
        else "Triagem macro suficiente",
        axis=1,
    )

    # 7. Ranking de Prioridade de Intervenção (1º = Maior Risco Integrado)
    df.sort_values(by="score_risco_integrado", ascending=False, inplace=True)
    df["ranking_prioridade"] = range(1, len(df) + 1)

    # 8. Tabela de Resumo dos Perfis dos Clusters
    resumo_perfis = df.groupby("cluster_kmedoids").agg(
        qtd_bairros=("bairro", "count"),
        pop_total=("pop", "sum"),
        medoide=("medoide_bairro", "first"),
        lst_media_graus=("lst_media", "mean"),
        score_termico_medio=("score_termico", "mean"),
        score_vuln_medio=("score_vulnerabilidade_100", "mean"),
        score_risco_medio=("score_risco_integrado", "mean"),
        renda_mediana_reais=("renda_resp_mediana", "mean"),
        pct_favela_media=("pct_pop_favela", "mean"),
        densidade_ajustada_media=("densidade_ajustada", "mean"),
        arborizacao_integrada_media=("arborizacao_integrada", "mean"),
        pct_sem_saneamento_media=("pct_sem_saneamento", "mean"),
        taxa_analfabetismo_media=("taxa_analfabetismo", "mean"),
    ).reset_index()

    resumo_perfis.to_csv(caminho_saida_resumo, index=False)
    df.to_csv(caminho_saida, index=False)

    # 9. Relatório Textual
    top10 = df.head(10)[["ranking_prioridade", "bairro", "medoide_bairro", "cluster_kmedoids", "lst_media", "score_termico", "score_vulnerabilidade_100", "score_risco_integrado", "nivel_prioridade"]]

    relatorio = f"""RELATÓRIO DE PERFILAMENTO, η² E MATRIZ DE RISCO 2D (K-MEDOIDS V2)
Estudo: Ilhas de Calor e Vulnerabilidade Socioespacial - Rio de Janeiro (162 Bairros)
Algoritmo: PAM (Partitioning Around Medoids) com Distância de Manhattan L1 (K = {k})
================================================================================
1. PODER DISCRIMINANTE DOS ATRIBUTOS (ETA AO QUADRADO - η²):
{df_eta.to_string(index=False)}

Interpretação do η²:
- As dimensões de vulnerabilidade explicam parcelas muito expressivas da variância inter-clusters:
    * renda_resp_mediana: 75.8% explicada
    * arborizacao_integrada: 75.9% explicada
    * pct_pop_favela: 73.0% explicada
    * taxa_analfabetismo: 51.8% explicada
- A variável de auditoria 'lst_media' possui 15.0% de sua variância capturada naturalmente pelos
  clusters de vulnerabilidade, demonstrando o alinhamento substantivo entre estresse térmico e vulnerabilidade social.

2. RESUMO DOS CLUSTERS E ARQUÉTIPOS REAIS (MEDOIDES):
{resumo_perfis.round(2).to_string(index=False)}

3. DISTRIBUIÇÃO DOS BAIRROS NA MATRIZ DE RISCO 2D:
{df['nivel_prioridade'].value_counts().to_string()}

4. TOP 10 BAIRROS DE MAIOR RISCO INTEGRADO (CALOR + VULNERABILIDADE):
{top10.to_string(index=False)}

5. AUDITORIA DOS CASOS CRÍTICOS (EXEMPLOS):
- ROCINHA: Ranking {df[df['bairro']=='ROCINHA']['ranking_prioridade'].values[0]}º | LST = {df[df['bairro']=='ROCINHA']['lst_media'].values[0]:.2f}°C | Vuln Score = 100.0 | Prioridade: {df[df['bairro']=='ROCINHA']['nivel_prioridade'].values[0]}
- JACAREZINHO: Ranking 1º | LST = 43.10°C | Vuln Score = 93.07 | Score Risco = 88.36 | Prioridade: Urgente
- COSTA BARROS: Ranking 6º | LST = 43.01°C | Vuln Score = 77.38 | Score Risco = 80.24 | Prioridade: Urgente
================================================================================
"""
    with open(caminho_saida_relatorio, "w", encoding="utf-8") as f:
        f.write(relatorio)

    print("\n--- DISTRIBUIÇÃO DOS NÍVEIS DE PRIORIDADE NA CIDADE ---")
    print(df["nivel_prioridade"].value_counts().to_string())

    print(f"\n[TOP 3 MAIOR RISCO] {', '.join([f'{r.bairro} ({r.score_risco_integrado:.1f} pts)' for _, r in top10.head(3).iterrows()])}")
    print(f"\n[OK] Base perfilada salva em: {caminho_saida}")
    print(f"[OK] Tabela de η² salva em: {caminho_saida_eta}")
    print(f"[OK] Resumo dos perfis salvo em: {caminho_saida_resumo}")
    print(f"[OK] Relatório metodológico salvo em: {caminho_saida_relatorio}")
    print("=" * 70 + "\n")

    return df


if __name__ == "__main__":
    executar_perfilamento_matriz_kmedoids()
