"""Gera dois datasets a partir do resultado final do K-Medoids (K=4).

  A) dataset_kmedoids_spearman.csv  -> Spearman, objetivo: analisar REDUNDANCIA
  B) dataset_kmedoids_eta2.csv      -> eta ao quadrado, objetivo: analisar RELEVANCIA

Uso (qualquer computador, so precisa de pandas e numpy):
    python gerar_datasets_selecao_variaveis.py
    python gerar_datasets_selecao_variaveis.py caminho/do/classificacao_bairros_ilhas_calor_kmedoids.csv
Saida: pasta compartilhada do Drive sprint3 (precisa estar sincronizada
pelo Google Drive para computador). Se nao for encontrada, grava em Dados_tratados/ ao lado
do script. Para forcar outra pasta: defina a variavel de ambiente PASTA_SAIDA.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

NOME_CSV = 'classificacao_bairros_ilhas_calor_kmedoids.csv'
LIMIAR_SPEARMAN = 0.70   # |rho| >= limiar => par redundante
LIMIAR_ETA2 = 0.14       # eta2 >= 0.14 => efeito grande (Cohen)

COLS_ID = ['codbairro', 'bairro', 'regiao_administrativa', 'cluster_id', 'cluster_nome']
# Selecao feita so sobre os 7 indicadores de vulnerabilidade usados no K-Medoids.
# Scores, ranking e distancias ficam de fora: derivam dos clusters.
CANDIDATAS = ['renda_resp_mediana', 'pct_pop_favela', 'densidade_ajustada',
              'arborizacao_integrada', 'pct_sem_saneamento', 'pct_pavimentacao',
              'taxa_analfabetismo']
# lst_media e CONTROLE, como no K-Medoids: fora do agrupamento e fora da selecao
# (nao e testada nem removida); entra nos arquivos so como coluna de validacao.
CONTROLE = 'lst_media'

AQUI = os.path.dirname(os.path.abspath(__file__))
ID_PASTA_DRIVE = '1DybSqPJtjEc-F2bDvY9ZFuUzFLr2C_rZ'   # drive.google.com/drive/folders/<id>


def achar_pasta_saida():
    """Pasta do Drive compartilhado sincronizada localmente; senao, Dados_tratados/."""
    if os.environ.get('PASTA_SAIDA'):
        return os.environ['PASTA_SAIDA']
    padroes = [f'{letra}:/.shortcut-targets-by-id/{ID_PASTA_DRIVE}/*' for letra in 'DEFGHIJKLMNOPQRSTUVWXYZ']
    padroes.append(os.path.expanduser(
        f'~/Library/CloudStorage/GoogleDrive-*/.shortcut-targets-by-id/{ID_PASTA_DRIVE}/*'))
    for padrao in padroes:
        achados = sorted(d for d in glob.glob(padrao) if os.path.isdir(d))
        if achados:
            return next((d for d in achados if os.path.basename(d) == 'sprint3'), achados[0])
    print('Aviso: pasta do Drive nao encontrada neste computador; gravando em Dados_tratados/.')
    return os.path.join(AQUI, 'Dados_tratados')


PASTA_SAIDA = achar_pasta_saida()


def achar_csv():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for raiz in (AQUI, os.getcwd(), os.path.join(os.path.expanduser('~'), 'Downloads')):
        achados = glob.glob(os.path.join(raiz, '**', NOME_CSV), recursive=True)
        if achados:
            return achados[0]
    raise FileNotFoundError(f'{NOME_CSV} nao encontrado. Passe o caminho como argumento.')


def eta_quadrado(x, grupos):
    ok = x.notna()                      # ausentes fora de TODAS as somas (len(g) contava NaN e mean() nao)
    x, grupos = x[ok], grupos[ok]
    m = x.mean()
    ss_total = ((x - m) ** 2).sum()
    ss_entre = sum(len(g) * (g.mean() - m) ** 2 for _, g in x.groupby(grupos))
    return ss_entre / ss_total


def selecao_spearman(X):
    rho = X.corr(method='spearman')
    mantidas, removidas = list(X.columns), []
    while True:
        sub = np.nan_to_num(rho.loc[mantidas, mantidas].abs().to_numpy(copy=True))  # NaN (coluna constante) = sem correlacao
        np.fill_diagonal(sub, 0)
        if sub.max() < LIMIAR_SPEARMAN:
            break
        a, b = np.unravel_index(sub.argmax(), sub.shape)
        par = [mantidas[a], mantidas[b]]
        medias = {v: sub[mantidas.index(v)].mean() for v in par}
        descarte = max(par, key=medias.get)
        outro = par[0] if descarte == par[1] else par[1]
        removidas.append((descarte, outro, float(rho.loc[par[0], par[1]])))
        mantidas.remove(descarte)
    return mantidas, removidas


def main():
    caminho = achar_csv()
    print('Lendo:', caminho)
    df = pd.read_csv(caminho, dtype={'codbairro': str}, encoding='utf-8-sig')
    df['codbairro'] = df['codbairro'].str.zfill(3)
    print(f'{df.shape[0]} linhas x {df.shape[1]} colunas | '
          f'clusters: {df["cluster_id"].value_counts().sort_index().to_dict()}')

    X = df[CANDIDATAS].apply(pd.to_numeric, errors='coerce').astype(float)
    if X.isna().any().any():
        print('Aviso: ha valores ausentes nas candidatas:', X.isna().sum()[X.isna().any()].to_dict())

    # A) Spearman
    mant_sp, remov_sp = selecao_spearman(X)
    print(f'\n[A] Spearman (|rho| >= {LIMIAR_SPEARMAN})')
    for v, outro, r in remov_sp:
        print(f'  removida {v} (redundante com {outro}, rho = {r:+.2f})')
    print('  mantidas:', mant_sp)
    if len(mant_sp) != 5:
        print(f'Aviso: o plano espera 5 variaveis por selecao, Spearman manteve {len(mant_sp)}.')

    # B) eta ao quadrado
    eta2 = pd.Series({c: eta_quadrado(X[c], df['cluster_id']) for c in CANDIDATAS}) \
             .sort_values(ascending=False)
    mant_eta = eta2[eta2 >= LIMIAR_ETA2].index.tolist()
    print(f'\n[B] eta2 (>= {LIMIAR_ETA2})')
    print(eta2.round(4).to_string())
    print('  mantidas:', mant_eta)
    if len(mant_eta) != 5:
        print(f'Aviso: o plano espera 5 variaveis por selecao, eta2 manteve {len(mant_eta)}.')
    eta_ctrl = eta_quadrado(pd.to_numeric(df[CONTROLE], errors='coerce'), df['cluster_id'])
    print(f'  controle {CONTROLE}: eta2 = {eta_ctrl:.4f} (so validacao, fora da selecao)')

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    saidas = {
        'dataset_kmedoids_spearman.csv': df[COLS_ID + mant_sp + [CONTROLE]],
        'dataset_kmedoids_eta2.csv': df[COLS_ID + mant_eta + [CONTROLE]],
    }
    print()
    for nome, d in saidas.items():
        destino = os.path.join(PASTA_SAIDA, nome)
        d.to_csv(destino, index=False, encoding='utf-8')
        print(f'Salvo: {destino} ({d.shape[0]} x {d.shape[1]})')


if __name__ == '__main__':
    main()
