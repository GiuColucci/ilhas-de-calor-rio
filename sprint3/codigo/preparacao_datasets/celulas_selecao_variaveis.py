# ==============================================================================
# CÉLULA 1 (markdown) — cole como texto:
#
# ## Dataset pós K-Medoids e seleção de variáveis (Sprint 2)
# Base: `classificacao_bairros_ilhas_calor_kmedoids.csv` (162 bairros com `cluster_id`).
# - **Dataset A — Spearman (redundância):** remove variáveis fortemente correlacionadas entre si.
# - **Dataset B — η² (relevância):** mantém variáveis que mais separam os clusters.
# ==============================================================================


# ==============================================================================
# CÉLULA 2 (código) — carregamento do dataset pós K-Medoids
# ==============================================================================
import os, glob
import numpy as np
import pandas as pd

def achar_arquivo(nome):
    """Procura o arquivo no Drive (Colab) ou em pastas locais (qualquer computador)."""
    raizes = ['/content/drive/MyDrive', '/content/drive/Shareddrives',
              '/content/drive/.shortcut-targets-by-id', '/content',
              os.getcwd(), os.path.expanduser('~')]
    try:
        raizes.insert(0, BASE_DIR)          # definido nas células anteriores do notebook
    except NameError:
        pass
    for raiz in raizes:
        if not os.path.exists(raiz):
            continue
        achados = glob.glob(os.path.join(raiz, '**', nome), recursive=True)
        if achados:
            return achados[0]
    raise FileNotFoundError(
        f'{nome} não encontrado. No Colab, confira se a pasta do projeto está como '
        'atalho em "Meu Drive"; fora do Colab, coloque o arquivo na pasta do notebook.')

CAMINHO_KM = achar_arquivo('classificacao_bairros_ilhas_calor_kmedoids.csv')
print('Arquivo encontrado em:', CAMINHO_KM)

df_km = pd.read_csv(CAMINHO_KM, dtype={'codbairro': str}, encoding='utf-8-sig')
df_km['codbairro'] = df_km['codbairro'].str.zfill(3)
print(f'Dimensões: {df_km.shape[0]} linhas x {df_km.shape[1]} colunas')
print('Bairros por cluster:', df_km['cluster_id'].value_counts().sort_index().to_dict())

# Colunas de identificação/cluster mantidas nos datasets finais
COLS_ID = ['codbairro', 'bairro', 'regiao_administrativa', 'cluster_id', 'cluster_nome']

# Candidatas: os 7 indicadores de vulnerabilidade do K-Medoids + LST (controle térmico).
# Scores, ranking e distâncias ficam de fora (derivam dos clusters => circularidade).
CANDIDATAS = ['renda_resp_mediana', 'pct_pop_favela', 'densidade_ajustada',
              'arborizacao_integrada', 'pct_sem_saneamento', 'pct_pavimentacao',
              'taxa_analfabetismo', 'lst_media']

X = df_km[CANDIDATAS].apply(pd.to_numeric, errors='coerce').astype(float)
print('Ausentes nas candidatas:', int(X.isna().sum().sum()))


# ==============================================================================
# CÉLULA 3 (código) — Dataset A: Spearman (objetivo: analisar REDUNDÂNCIA)
# ==============================================================================
LIMIAR_SPEARMAN = 0.70   # |rho| >= limiar => par redundante (ajuste se quiser)

rho = X.corr(method='spearman')

# Poda gulosa: enquanto houver par redundante, remove a variável com maior |rho| médio
# em relação às demais (a mais "explicada" pelo resto da base).
mantidas = list(CANDIDATAS)
removidas_sp = []
while True:
    sub = rho.loc[mantidas, mantidas].abs().copy()
    vals = sub.to_numpy(copy=True)
    np.fill_diagonal(vals, 0)
    sub = pd.DataFrame(vals, index=sub.index, columns=sub.columns)
    if vals.max() < LIMIAR_SPEARMAN:
        break
    a, b = np.unravel_index(vals.argmax(), vals.shape)
    par = [sub.index[a], sub.columns[b]]
    descarte = max(par, key=lambda v: sub[v].mean())
    removidas_sp.append((descarte, [p for p in par if p != descarte][0],
                         float(rho.loc[par[0], par[1]])))
    mantidas.remove(descarte)

print(f'Limiar |rho| >= {LIMIAR_SPEARMAN}')
for v, par, r in removidas_sp:
    print(f'  removida: {v:<24} (redundante com {par}, rho = {r:+.2f})')
print('Mantidas (Spearman):', mantidas)

df_spearman = df_km[COLS_ID + mantidas].copy()
print(f'df_spearman: {df_spearman.shape[0]} linhas x {df_spearman.shape[1]} colunas')


# ==============================================================================
# CÉLULA 4 (código) — Dataset B: η² (objetivo: analisar RELEVÂNCIA)
# ==============================================================================
LIMIAR_ETA2 = 0.14       # convenção de Cohen: η² >= 0.14 = efeito grande

def eta_quadrado(x, grupos):
    """η² = SS_entre / SS_total de uma variável contínua pelos clusters."""
    m = x.mean()
    ss_total = ((x - m) ** 2).sum()
    ss_entre = sum(len(g) * (g.mean() - m) ** 2 for _, g in x.groupby(grupos))
    return ss_entre / ss_total

eta2 = pd.Series({c: eta_quadrado(X[c], df_km['cluster_id']) for c in CANDIDATAS}) \
         .sort_values(ascending=False)
print('η² por atributo (cluster_id):')
print(eta2.round(4).to_string())

mantidas_eta = eta2[eta2 >= LIMIAR_ETA2].index.tolist()
print(f'\nMantidas (η² >= {LIMIAR_ETA2}):', mantidas_eta)
print('Removidas:', [c for c in CANDIDATAS if c not in mantidas_eta])

df_eta2 = df_km[COLS_ID + mantidas_eta].copy()
print(f'df_eta2: {df_eta2.shape[0]} linhas x {df_eta2.shape[1]} colunas')


# ==============================================================================
# CÉLULA 5 (código) — salvar os dois datasets (só adiciona arquivos novos)
# ==============================================================================
try:
    PASTA_SAIDA = os.path.join(BASE_DIR, 'data', 'processed')
except NameError:
    PASTA_SAIDA = os.getcwd()
os.makedirs(PASTA_SAIDA, exist_ok=True)

caminho_sp  = os.path.join(PASTA_SAIDA, 'dataset_kmedoids_spearman.csv')
caminho_eta = os.path.join(PASTA_SAIDA, 'dataset_kmedoids_eta2.csv')
df_spearman.to_csv(caminho_sp,  index=False, encoding='utf-8')
df_eta2.to_csv(caminho_eta, index=False, encoding='utf-8')

print('Salvos em:')
print(' ', caminho_sp)
print(' ', caminho_eta)
display(df_spearman.head())
display(df_eta2.head())
