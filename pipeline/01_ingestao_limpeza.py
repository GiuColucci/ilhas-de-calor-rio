"""
ETAPA 1: INGESTÃO, AUDITORIA E LIMPEZA DE DADOS (K-MEDOIDS V2)
Ciclo de Ciência de Dados - Agrupamento por Vulnerabilidade Socioambiental e Suscetibilidade Térmica
Município do Rio de Janeiro (162 Bairros)

Objetivo:
- Carregar a base de dados estática por bairro (output/base_bairro_estatica.csv).
- Auditar a completude cadastral e integridade das variáveis socioeconômicas, de infraestrutura e ambientais.
- Tratar valores ausentes pontuais (imputação de 'pct_sem_agua_rede' no bairro Joá via mediana da RA 24).
- Auditar as novas variáveis de vulnerabilidade (taxa_analfabetismo, pct_sem_esgoto_rede).
- Padronizar nomenclaturas e exportar a base saneada em 'outputs/dados_limpos.csv'.
"""

from pathlib import Path
import pandas as pd


def executar_ingestao_limpeza() -> pd.DataFrame:
    # 1. Definição de caminhos dinâmicos e resilientes
    diretorio_atual = Path(__file__).resolve().parent
    diretorio_raiz = diretorio_atual
    while diretorio_raiz != diretorio_raiz.parent:
        if (diretorio_raiz / 'data').exists() or (diretorio_raiz / 'output').exists():
            break
        diretorio_raiz = diretorio_raiz.parent

    # Busca resiliente da base de entrada (funciona se a pasta estiver isolada no Google Drive ou no repo)
    candidatos_entrada = [
        diretorio_atual / "dados" / "base_bairro_estatica.csv",
        diretorio_atual / "base_bairro_estatica.csv",
        diretorio_raiz / "output" / "base_bairro_estatica.csv",
    ]
    caminho_entrada = next((p for p in candidatos_entrada if p.exists()), candidatos_entrada[0])
    caminho_saida_dir = diretorio_atual / "outputs"
    caminho_saida_csv = caminho_saida_dir / "dados_limpos.csv"

    caminho_saida_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ETAPA 1: INGESTÃO, AUDITORIA & LIMPEZA DE DADOS (K-MEDOIDS V2)")
    print("=" * 70)

    # 2. Ingestão da base
    if not caminho_entrada.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado em: {candidatos_entrada}")

    df = pd.read_csv(caminho_entrada)
    print(f"[OK] Base bruta carregada de: {caminho_entrada}")
    print(f"     Dimensões: {df.shape[0]} bairros e {df.shape[1]} colunas.")

    # 3. Auditoria geral de valores ausentes
    nulos = df.isnull().sum()
    colunas_com_nulo = nulos[nulos > 0]

    if not colunas_com_nulo.empty:
        print("\n[AUDITORIA] Valores ausentes detectados:")
        for col, qtd in colunas_com_nulo.items():
            bairros_afetados = df[df[col].isnull()]["bairro"].tolist()
            print(f"  - Coluna '{col}': {qtd} valor(es) nulo(s) no(s) bairro(s): {bairros_afetados}")

        # 4. Tratamento do valor ausente pontual em pct_sem_agua_rede (Joá / RA 24 - Barra da Tijuca)
        for idx in df[df["pct_sem_agua_rede"].isnull()].index:
            ra_codigo = df.loc[idx, "cod_ra"]
            bairro_nome = df.loc[idx, "bairro"]

            mediana_ra = df[df["cod_ra"] == ra_codigo]["pct_sem_agua_rede"].median()
            valor_imputado = mediana_ra if pd.notnull(mediana_ra) else df["pct_sem_agua_rede"].median()

            df.loc[idx, "pct_sem_agua_rede"] = valor_imputado
            print(f"[TRATAMENTO] 'pct_sem_agua_rede' em '{bairro_nome}' imputado com {valor_imputado:.4f} (mediana da RA {ra_codigo}).")
    else:
        print("\n[AUDITORIA] Nenhum valor ausente encontrado na base bruta.")

    # 5. Auditoria de integridade das variáveis essenciais do modelo V2
    colunas_chave = [
        "renda_resp_mediana",
        "pct_pop_favela",
        "densidade_ajustada",
        "pct_sem_arborizacao",
        "pct_arb_densa",
        "pct_sem_agua_rede",
        "pct_sem_esgoto_rede",
        "pct_pavimentacao",
        "taxa_analfabetismo",
        "lst_media"
    ]
    print("\n[AUDITORIA] Checagem das variáveis-chave de modelagem V2:")
    for col in colunas_chave:
        if col in df.columns:
            n_nulos = df[col].isnull().sum()
            print(f"  ✓ {col:<23}: {df[col].count()} preenchidos | {n_nulos} nulos | min: {df[col].min():.2f} | max: {df[col].max():.2f}")
        else:
            print(f"  ✗ ALERTA: Coluna '{col}' ausente no dataset!")

    # 6. Padronização cadastral de strings
    df["bairro"] = df["bairro"].astype(str).str.strip().str.upper()
    df["regiao_administrativa"] = df["regiao_administrativa"].astype(str).str.strip()

    # 7. Exportação dos dados limpos
    df.to_csv(caminho_saida_csv, index=False)
    print(f"\n[OK] Base limpa e saneada exportada com sucesso em:\n     -> {caminho_saida_csv}")
    print("=" * 70 + "\n")

    return df


if __name__ == "__main__":
    executar_ingestao_limpeza()
