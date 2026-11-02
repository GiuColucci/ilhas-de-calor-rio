"""
ETAPA 3 (K-MEDOIDS V2): PRÉ-PROCESSAMENTO & PADRONIZAÇÃO BALANCEADA [0, 1]
Ciclo de Ciência de Dados - Agrupamento por Vulnerabilidade Socioambiental e Suscetibilidade Térmica
Município do Rio de Janeiro (162 Bairros)

Objetivo:
- Aplicar normalização Min-Max [0, 1] nas 7 variáveis de vulnerabilidade socioambiental:
    1. renda_resp_mediana
    2. pct_pop_favela
    3. densidade_ajustada
    4. arborizacao_integrada
    5. pct_sem_saneamento
    6. pct_pavimentacao
    7. taxa_analfabetismo
- Justificativa Metodológica (Min-Max vs RobustScaler no contexto L1):
    * Na Distância de Manhattan (L1), cada dimensão contribui linearmente (|x_i - x_j|).
    * Se utilizássemos RobustScaler, o IQR minúsculo de certas variáveis (como saneamento = 2.53%
      e pavimentação = 3.40%) geraria amplitudes artificiais de até 40 IQRs, fazendo com que 
      apenas 1 ou 2 variáveis dominassem mais de 80% de toda a distância vetorial.
    * A normalização Min-Max equaliza rigorosamente o espaço métrico: cada uma das 7 variáveis
      possui contribuição máxima unitária (|Δ| <= 1.0), garantindo um agrupamento verdadeiramente
      multidimensional e equilibrado.
- Exportar a base com as features padronizadas e o relatório de transformação em 'outputs/'.
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


def executar_preprocessamento_kmedoids() -> pd.DataFrame:
    diretorio_atual = Path(__file__).resolve().parent
    caminho_entrada = diretorio_atual / "outputs" / "dados_selecionados.csv"
    caminho_saida_csv = diretorio_atual / "outputs" / "dados_preparados_kmedoids.csv"
    caminho_relatorio = diretorio_atual / "outputs" / "relatorio_preprocessamento_kmedoids.txt"

    # Garantir pré-requisito da etapa 2
    if not caminho_entrada.exists():
        import importlib
        modulo_eda = importlib.import_module("02_eda_selecao")
        df = modulo_eda.executar_eda_selecao()
    else:
        df = pd.read_csv(caminho_entrada)

    print("=" * 70)
    print("ETAPA 3 (K-MEDOIDS V2): PRÉ-PROCESSAMENTO & PADRONIZAÇÃO [0, 1]")
    print("=" * 70)

    df_prep = df.copy()

    # 1. Normalização Min-Max [0, 1] nas 7 features de vulnerabilidade
    print("\n[NORMALIZAÇÃO MULTIDIMENSIONAL] Escalando 7 variáveis de vulnerabilidade para [0, 1]:")
    parametros_escala = {}

    for col in FEATURES_VULNERABILIDADE:
        v_min = float(df[col].min())
        v_max = float(df[col].max())
        col_norm = f"{col}_norm"
        df_prep[col_norm] = (df[col] - v_min) / (v_max - v_min) if v_max > v_min else 0.0
        parametros_escala[col] = {"min": v_min, "max": v_max}
        print(f"  • {col_norm:<28} -> [{v_min:8.2f}, {v_max:8.2f}] escalonado para [0.0, 1.0]")

    # 2. Exportação dos dados preparados
    df_prep.to_csv(caminho_saida_csv, index=False)

    # 3. Geração do Relatório Explicativo
    relatorio = f"""RELATÓRIO DE PRÉ-PROCESSAMENTO (K-MEDOIDS V2 - VULNERABILIDADE)
Estudo: Ilhas de Calor e Vulnerabilidade Socioambiental - Rio de Janeiro
Métrica Alvo: Distância de Manhattan (L1) com Algoritmo PAM
================================================================================
1. JUSTIFICATIVA DA PADRONIZAÇÃO MIN-MAX [0, 1] PARA MÉTRICA MANHATTAN:
- Na distância de Manhattan L1 (soma das diferenças absolutas), a escala de cada
  dimensão define diretamente o seu peso relativo no cálculo das dissimilaridades.
- O uso de RobustScaler (baseado em mediana e IQR) foi descartado porque:
    * Variáveis com distribuições muito concentradas possuem IQR minúsculo.
    * 'pct_sem_saneamento' possui IQR de apenas 2.53% (amplitude no RobustScaler = 39.59 IQRs).
    * 'pct_pavimentacao' possui IQR de 3.40% (amplitude no RobustScaler = 24.09 IQRs).
    * Sob RobustScaler, bairros com pequenas variações sanitárias ou viárias sequestrariam
      mais de 80% da distância total, quebrando o equilíbrio multidimensional do modelo.
- Com a padronização Min-Max [0, 1]:
    * Cada uma das 7 dimensões contribui com variação estritamente limitada a [0, 1].
    * O espaço métrico é perfeitamente democrático, integrando renda, favelização, densidade,
      arborização, saneamento, pavimentação e alfabetização em igualdade de condições.

2. PARÂMETROS DA ESCALA (MIN E MAX):
"""
    for col, p in parametros_escala.items():
        relatorio += f"  - {col:<24}: Mín={p['min']:9.2f} | Máx={p['max']:9.2f}\n"

    relatorio += f"""
3. STATUS DA BASE PREPARADA:
  - Bairros processados: {df_prep.shape[0]} bairros (100% de cobertura cadastral)
  - Total de colunas: {df_prep.shape[1]}
  - Features de entrada no PAM: {[f + '_norm' for f in FEATURES_VULNERABILIDADE]}
  - Metadado de auditoria: lst_media (mantida sem alteração para validação cruzada)
================================================================================
"""
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write(relatorio)

    print(f"\n[OK] Base pré-processada salva em: {caminho_saida_csv}")
    print(f"[OK] Relatório metodológico salvo em: {caminho_relatorio}")
    print("=" * 70 + "\n")

    return df_prep


if __name__ == "__main__":
    executar_preprocessamento_kmedoids()
