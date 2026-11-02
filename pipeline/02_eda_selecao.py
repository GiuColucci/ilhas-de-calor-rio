"""
ETAPA 2: ANÁLISE EXPLORATÓRIA (EDA) & SELEÇÃO DE FEATURES (K-MEDOIDS V2)
Ciclo de Ciência de Dados - Agrupamento por Vulnerabilidade Socioambiental e Suscetibilidade Térmica
Município do Rio de Janeiro (162 Bairros)

Objetivo:
- Estruturar o espaço multidimensional de 7 variáveis de vulnerabilidade e suscetibilidade urbana:
    1. renda_resp_mediana: Capacidade adaptativa econômico-financeira (R$).
    2. pct_pop_favela: Exposição habitacional precária / assentamentos informais (%).
    3. densidade_ajustada: Densidade demográfica na mancha habitada (hab/km² - trata MAUP).
    4. arborizacao_integrada: Infraestrutura verde mitigadora aglutinada (0 a 100%).
    5. pct_sem_saneamento: Déficit agregado de saneamento básico [água + esgoto] (0 a 100%).
    6. pct_pavimentacao: Grau de impermeabilização e artificialização do solo (%).
    7. taxa_analfabetismo: Vulnerabilidade educacional e capital social de resposta (%).
- Tratar as limitações metodológicas:
    * Aglutinação de arborização: elimina a multicolinearidade severa (r = -0.898).
    * Aglutinação de saneamento: unifica água e esgoto em indicador sintético robusto.
    * Desacoplamento da LST: 'lst_media' é reservada como metadado para validação cruzada do calor orbital.
    * Remoção de faixas etárias extremas (60+ e 0-4): evita dispersão demográfica irrelevante.
- Exportar a base com as features selecionadas e o relatório analítico detalhado.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# 1. Definição formal das colunas
METADADOS = [
    "codbairro",
    "bairro",
    "regiao_administrativa",
    "cod_ra",
    "pop",
    "area_km2",
    "lst_media"  # Mantida estritamente como metadado de auditoria e validação térmica
]

FEATURES_VULNERABILIDADE = [
    "renda_resp_mediana",      # Capacidade adaptativa financeira (renda mediana dos responsáveis)
    "pct_pop_favela",          # Vulnerabilidade habitacional e morfologia informal
    "densidade_ajustada",      # Concentração humana na área urbanizada (hab/km² domiciliado)
    "arborizacao_integrada",   # Índice sintético de arborização urbana [100 - sem_arb + arb_densa]/2
    "pct_sem_saneamento",      # Déficit sanitário agregado [sem água + sem esgoto]/2
    "pct_pavimentacao",        # Grau de artificialização / impermeabilização do solo
    "taxa_analfabetismo"       # Vulnerabilidade educacional / capital humano
]


def executar_eda_selecao() -> pd.DataFrame:
    diretorio_atual = Path(__file__).resolve().parent
    caminho_entrada = diretorio_atual / "outputs" / "dados_limpos.csv"
    caminho_saida_csv = diretorio_atual / "outputs" / "dados_selecionados.csv"
    caminho_relatorio = diretorio_atual / "outputs" / "relatorio_eda_selecao.txt"

    # Garantir que a Etapa 1 foi executada
    if not caminho_entrada.exists():
        import importlib
        modulo_ingestao = importlib.import_module("01_ingestao_limpeza")
        df = modulo_ingestao.executar_ingestao_limpeza()
    else:
        df = pd.read_csv(caminho_entrada)

    print("=" * 70)
    print("ETAPA 2: EDA & ENGENHARIA DE FEATURES (K-MEDOIDS V2 - VULNERABILIDADE)")
    print("=" * 70)

    # 2. Engenharia de Atributos (Feature Engineering & Aglutinações)
    # A) Arborização Integrada: resolve a colinearidade entre ausência e densidade arbórea
    df["arborizacao_integrada"] = ((100.0 - df["pct_sem_arborizacao"]) + df["pct_arb_densa"]) / 2.0

    # B) Saneamento Básico Integrado: média do déficit de água encanada e esgotamento sanitário
    df["pct_sem_saneamento"] = (df["pct_sem_agua_rede"] + df["pct_sem_esgoto_rede"]) / 2.0

    print("[ENGENHARIA DE ATRIBUTOS REALIZADA]")
    print("  ✓ 'arborizacao_integrada' criada via média de presença verde e densidade vegetal.")
    print("  ✓ 'pct_sem_saneamento' criada via média do déficit de água e esgotamento sanitário.")
    print("  ✓ 'lst_media' mantida como metadado para avaliação ex-post da eficácia dos clusters.")

    # 3. Estatísticas Descritivas das 7 Features de Vulnerabilidade
    stats_df = df[FEATURES_VULNERABILIDADE].describe().T
    stats_df["skew"] = df[FEATURES_VULNERABILIDADE].skew()

    print("\n[PANORAMA DAS 7 VARIÁVEIS DE VULNERABILIDADE SELECIONADAS]")
    for feat in FEATURES_VULNERABILIDADE:
        s = df[feat]
        print(f"  • {feat:<22} | Média: {s.mean():>9.2f} | Mediana: {s.median():>9.2f} | Mín: {s.min():>8.2f} | Máx: {s.max():>8.2f} | Assimetria: {s.skew():>+.2f}")

    # 4. Diagnóstico de Multicolinearidade
    matriz_corr = df[FEATURES_VULNERABILIDADE].corr()
    print("\n[MATRIZ DE CORRELAÇÃO DE PEARSON (7 FEATURES)]")
    print(matriz_corr.round(3).to_string())

    max_corr_par = None
    max_corr_val = 0.0
    for i in range(len(FEATURES_VULNERABILIDADE)):
        for j in range(i + 1, len(FEATURES_VULNERABILIDADE)):
            val = abs(matriz_corr.iloc[i, j])
            if val > max_corr_val:
                max_corr_val = val
                max_corr_par = (FEATURES_VULNERABILIDADE[i], FEATURES_VULNERABILIDADE[j], matriz_corr.iloc[i, j])

    print(f"\n[DIAGNÓSTICO DE COLINEARIDADE] Maior correlação observada: {max_corr_par[0]} x {max_corr_par[1]} = {max_corr_par[2]:+.3f}")
    if max_corr_val < 0.75:
        print("  ✓ Nenhuma correlação excessiva (|r| >= 0.75). Multicolinearidade crítica eliminada com sucesso!")
    else:
        print("  ⚠ Atenção: Presença de pares correlacionados acima de 0.75.")

    # 5. Associação com a Temperatura de Superfície (LST_MEDIA) para Auditoria
    corr_lst = df[FEATURES_VULNERABILIDADE].apply(lambda s: s.corr(df["lst_media"])).sort_values(ascending=False)
    print("\n[ASSOCIAÇÃO EXPLORATÓRIA COM A TEMPERATURA DE SUPERFÍCIE (LST_MEDIA)]")
    for var, r in corr_lst.items():
        print(f"  - {var:<22}: r = {r:+.3f}")

    # 6. Diagnóstico de Extremos e Outliers Morfológicos
    top_renda = df.nlargest(3, "renda_resp_mediana")[["bairro", "renda_resp_mediana", "pct_pop_favela"]]
    bot_renda = df.nsmallest(3, "renda_resp_mediana")[["bairro", "renda_resp_mediana", "pct_pop_favela"]]
    top_favela = df.nlargest(3, "pct_pop_favela")[["bairro", "pct_pop_favela", "taxa_analfabetismo"]]
    bot_pav = df.nsmallest(3, "pct_pavimentacao")[["bairro", "pct_pavimentacao", "pct_pop_favela"]]

    print("\n[DIAGNÓSTICO DE CONTRASTES URBANOS EXTREMOS]")
    print(f"  • Top 3 Renda Mediana: {', '.join([f'{r.bairro} (R$ {r.renda_resp_mediana:.0f})' for _, r in top_renda.iterrows()])}")
    print(f"  • Bottom 3 Renda Mediana: {', '.join([f'{r.bairro} (R$ {r.renda_resp_mediana:.0f})' for _, r in bot_renda.iterrows()])}")
    print(f"  • Top 3 % Favela: {', '.join([f'{r.bairro} ({r.pct_pop_favela:.1f}%)' for _, r in top_favela.iterrows()])}")
    print(f"  • Outliers de Pavimentação: {', '.join([f'{r.bairro} ({r.pct_pavimentacao:.1f}%)' for _, r in bot_pav.iterrows()])}")

    # 7. Estruturação e Exportação do Dataset Final Selecionado
    colunas_finais = METADADOS + FEATURES_VULNERABILIDADE
    df_selecionado = df[colunas_finais].copy()
    df_selecionado.to_csv(caminho_saida_csv, index=False)

    # 8. Geração do Relatório Textual de Diagnóstico
    relatorio_texto = f"""RELATÓRIO DE EDA & SELEÇÃO DE FEATURES - K-MEDOIDS V2
Modelo: Agrupamento por Vulnerabilidade Socioambiental e Suscetibilidade Térmica
Município do Rio de Janeiro - 162 Bairros
======================================================================
1. OBJETIVO DO MODELO V2:
Agrupar os bairros cariocas com base nas dimensões de suscetibilidade intrínseca
e capacidade adaptativa frente ao estresse térmico urbano, superando o agrupamento
puramente biofísico do modelo V1 e do Ward.

2. AS 7 FEATURES SELECIONADAS:
- renda_resp_mediana: Capacidade financeira de mitigação/adaptação (R$).
- pct_pop_favela: Fragilidade construtiva e segregação socioespacial (%).
- densidade_ajustada: Adensamento humano na área efetivamente habitada (hab/km²).
- arborizacao_integrada: Infraestrutura verde consolidada [(100 - % sem arb) + % arb densa]/2 (%).
- pct_sem_saneamento: Déficit de infraestrutura sanitária básica [(sem água + sem esgoto)/2] (%).
- pct_pavimentacao: Grau de selamento do solo urbano (%).
- taxa_analfabetismo: Barreiras educacionais e acesso a redes de alerta (%).

3. TRATAMENTO DE LIMITAÇÕES REALIZADO:
- Multicolinearidade severa de arborização eliminada (r anterior = -0.898).
- Resolução da dependência espacial MAUP usando densidade ajustada.
- Desacoplamento da LST_media do espaço de agrupamento: utilizada como controle ex-post.
- Eliminação das variáveis 60+ e 0-4 anos para foco em vulnerabilidade estrutural.

4. ESTATÍSTICAS DESCRITIVAS:
{stats_df.round(2).to_string()}

5. MATRIZ DE CORRELAÇÃO DE PEARSON (7 FEATURES):
{matriz_corr.round(3).to_string()}

6. CORRELAÇÃO LINEAR COM LST_MEDIA (AUDITORIA FÍSICA):
{corr_lst.round(3).to_string()}

7. CONTRASTES URBANOS DESTACADOS:
- Renda Mediana: Varia de R$ {df['renda_resp_mediana'].min():.2f} ({bot_renda.iloc[0]['bairro']}) a R$ {df['renda_resp_mediana'].max():.2f} ({top_renda.iloc[0]['bairro']}).
- População em Favela: Varia de 0.0% a {df['pct_pop_favela'].max():.1f}% ({top_favela.iloc[0]['bairro']}).
- Arborização Integrada: Média de {df['arborizacao_integrada'].mean():.1f}%, variando de {df['arborizacao_integrada'].min():.1f}% a {df['arborizacao_integrada'].max():.1f}%.
- Déficit de Saneamento: Média de {df['pct_sem_saneamento'].mean():.1f}%, atingindo até {df['pct_sem_saneamento'].max():.1f}%.
======================================================================
"""
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write(relatorio_texto)

    print(f"\n[OK] Base selecionada salva em: {caminho_saida_csv}")
    print(f"[OK] Relatório analítico salvo em: {caminho_relatorio}")
    print("=" * 70 + "\n")

    return df_selecionado


if __name__ == "__main__":
    executar_eda_selecao()
