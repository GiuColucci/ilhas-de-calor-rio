"""
ETAPA 6 (K-MEDOIDS V2): GERAÇÃO DE OUTPUTS FINAIS E INTEGRAÇÃO (DINÂMICO PARA QUALQUER K)
Ciclo de Ciência de Dados - Agrupamento por Vulnerabilidade Socioambiental e Suscetibilidade Térmica
Município do Rio de Janeiro (162 Bairros)

Objetivo:
- Gerar os 2 outputs tabulares analíticos finais do modelo K-Medoids V2:
    1. Output 1: Arquivo Tabular Final (classificacao_bairros_ilhas_calor_kmedoids.csv) com os 162 bairros,
       contendo ranking de risco integrado, cluster_id, medoide arquetípico, escore de vulnerabilidade,
       escore térmico (LST), prioridade de intervenção e alerta de MAUP.
    2. Output 2: Tabela-Resumo dos Perfis dos Clusters (perfil_resumo_clusters_kmedoids.csv) adaptado
       dinamicamente para a quantidade de clusters K presente nos dados, com diretrizes estratégicas.
- Acionar a geração do Mapa Interativo HTML com seletor dinâmico de K e camada de confirmação
  visual da LST média observada por satélite.
"""

from pathlib import Path
import pandas as pd
import importlib


def obter_diretriz_estrategica(c_id: int, k: int) -> str:
    """Gera diretriz estratégica de planejamento urbano dinâmica para qualquer cluster e K."""
    if c_id == 0:
        return "Preservação rigorosa da cobertura florestal e áreas verdes, contenção da expansão urbana e fomento a corredores de ventilação natural."
    elif c_id == k - 1:
        return "Intervenção prioritária urgente: regularização de água/esgoto, micro-parques arborizados, telhados claros reflexivos e pontos comunitários de hidratação e socorro térmico."
    elif c_id == 1 and k >= 4:
        return "Mitigação severa da impermeabilização: substituição de asfalto/calçadas por pisos permeáveis, plantio massivo de árvores de sombra e incentivo a telhados verdes."
    elif c_id == k - 2 and k >= 4:
        return "Revitalização e arborização de praças públicas, qualificação da drenagem e saneamento local, e campanhas comunitárias de conscientização sobre riscos térmicos."
    else:
        return "Monitoramento microclimático contínuo, manutenção da arborização viária e proteção preventiva de praças e áreas verdes."


def executar_geracao_outputs_kmedoids() -> tuple[pd.DataFrame, pd.DataFrame]:
    diretorio_atual = Path(__file__).resolve().parent
    caminho_entrada = diretorio_atual / "outputs" / "dados_perfilados_matriz_kmedoids.csv"
    caminho_saida_dir = diretorio_atual / "outputs"

    caminho_output_1 = caminho_saida_dir / "classificacao_bairros_ilhas_calor_kmedoids.csv"
    caminho_output_2 = caminho_saida_dir / "perfil_resumo_clusters_kmedoids.csv"

    # Garantir pré-requisito da etapa 5
    if not caminho_entrada.exists():
        modulo_perfil = importlib.import_module("05_perfilamento_matriz_kmedoids")
        df = modulo_perfil.executar_perfilamento_matriz_kmedoids()
    else:
        df = pd.read_csv(caminho_entrada)

    clusters_unicos = sorted(df["cluster_kmedoids"].dropna().unique())
    k = len(clusters_unicos)

    print("=" * 70)
    print(f"ETAPA 6 (K-MEDOIDS V2): GERAÇÃO DE OUTPUTS FINAIS (K = {k} CLUSTERS)")
    print("=" * 70)

    # 1. OUTPUT 1: ARQUIVO TABULAR FINAL DOS 162 BAIRROS
    colunas_output_1 = [
        "ranking_prioridade",
        "bairro",
        "codbairro",
        "regiao_administrativa",
        "cluster_kmedoids",
        "cluster_nome",
        "medoide_bairro",
        "is_medoide",
        "dist_medoide_l1",
        "score_vulnerabilidade_100",
        "score_termico",
        "score_risco_integrado",
        "nivel_prioridade",
        "acao_recomendada",
        "alerta_maup",
        "lst_media",
        "renda_resp_mediana",
        "pct_pop_favela",
        "densidade_ajustada",
        "arborizacao_integrada",
        "pct_sem_saneamento",
        "pct_pavimentacao",
        "taxa_analfabetismo",
        "pop",
        "area_km2",
    ]

    colunas_existentes = [c for c in colunas_output_1 if c in df.columns]
    df_out1 = df[colunas_existentes].rename(columns={"cluster_kmedoids": "cluster_id"}).copy()
    df_out1.to_csv(caminho_output_1, index=False)
    print(f"[OK] Output 1 gerado: {caminho_output_1} ({len(df_out1)} bairros classificados)")

    # 2. OUTPUT 2: TABELA-RESUMO DOS PERFIS DOS CLUSTERS
    pop_total_cidade = df["pop"].sum()
    total_bairros = len(df)

    resumo = df.groupby(["cluster_kmedoids", "cluster_nome"]).agg(
        medoide=("medoide_bairro", "first"),
        qtd_bairros=("bairro", "count"),
        populacao_total=("pop", "sum"),
        score_vuln_medio=("score_vulnerabilidade_100", "mean"),
        score_termico_medio=("score_termico", "mean"),
        score_risco_medio=("score_risco_integrado", "mean"),
        lst_media_graus_c=("lst_media", "mean"),
        renda_mediana_reais=("renda_resp_mediana", "mean"),
        pct_pop_favela_media=("pct_pop_favela", "mean"),
        densidade_ajustada_media=("densidade_ajustada", "mean"),
        arborizacao_integrada_media=("arborizacao_integrada", "mean"),
        pct_sem_saneamento_media=("pct_sem_saneamento", "mean"),
        pct_pavimentacao_media=("pct_pavimentacao", "mean"),
        taxa_analfabetismo_media=("taxa_analfabetismo", "mean"),
    ).reset_index().rename(columns={"cluster_kmedoids": "cluster_id"})

    resumo["pct_bairros"] = (100.0 * resumo["qtd_bairros"] / total_bairros).round(1)
    resumo["pct_populacao_cidade"] = (100.0 * resumo["populacao_total"] / pop_total_cidade).round(1)
    resumo["diretriz_estrategica"] = resumo["cluster_id"].apply(lambda cid: obter_diretriz_estrategica(cid, k))

    colunas_decimais = [
        "score_vuln_medio",
        "score_termico_medio",
        "score_risco_medio",
        "lst_media_graus_c",
        "renda_mediana_reais",
        "pct_pop_favela_media",
        "densidade_ajustada_media",
        "arborizacao_integrada_media",
        "pct_sem_saneamento_media",
        "pct_pavimentacao_media",
        "taxa_analfabetismo_media",
    ]
    for col in colunas_decimais:
        resumo[col] = resumo[col].round(2)

    colunas_ordenadas_resumo = [
        "cluster_id",
        "cluster_nome",
        "medoide",
        "qtd_bairros",
        "pct_bairros",
        "populacao_total",
        "pct_populacao_cidade",
        "score_vuln_medio",
        "score_termico_medio",
        "score_risco_medio",
        "lst_media_graus_c",
        "renda_mediana_reais",
        "pct_pop_favela_media",
        "densidade_ajustada_media",
        "arborizacao_integrada_media",
        "pct_sem_saneamento_media",
        "pct_pavimentacao_media",
        "taxa_analfabetismo_media",
        "diretriz_estrategica",
    ]

    df_out2 = resumo[colunas_ordenadas_resumo]
    df_out2.to_csv(caminho_output_2, index=False)
    print(f"[OK] Output 2 gerado: {caminho_output_2} ({len(df_out2)} clusters sumarizados)")

    # Exibição no console
    print("\n--- RESUMO EXECUTIVO DOS OUTPUTS GERADOS (K-MEDOIDS V2) ---")
    print(df_out2[["cluster_id", "medoide", "qtd_bairros", "score_vuln_medio", "lst_media_graus_c", "score_risco_medio"]].to_string(index=False))

    # 3. Integração com a Geração do Mapa Interativo HTML e Figuras Estáticas
    try:
        modulo_mapa = importlib.import_module("gerar_mapa_html_kmedoids")
        modulo_mapa.gerar_mapa_html_kmedoids()
    except Exception as e:
        print(f"[AVISO] Geração do mapa HTML delegada ou com aviso: {e}")

    try:
        modulo_figuras = importlib.import_module("gerar_figuras_mapa_relatorio")
        modulo_figuras.gerar_figuras_estaticas_kmedoids()
    except Exception as e:
        print(f"[AVISO] Geração das figuras estáticas delegada ou com aviso: {e}")

    print("=" * 70 + "\n")
    return df_out1, df_out2


if __name__ == "__main__":
    executar_geracao_outputs_kmedoids()
