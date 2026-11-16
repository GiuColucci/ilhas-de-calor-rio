"""
ENG4040 - Projeto Integrado 4 (turma 33B) - Sprint 3
Comparacao de tecnicas de clustering nos dois subsets de vulnerabilidade.

Sem folds: cada cenario e ajustado nos 162 bairros. As metricas sao calculadas sobre
esse agrupamento. A estabilidade e medida pelo ARI bootstrap (100 reamostragens de
80% dos bairros, refeitas e comparadas com o agrupamento completo).

Cenarios (32 no total):
  - K-Medoids : 2 selecoes x K in {3,4,5} x distancia {Manhattan, Euclidiana}  -> 12
  - AGNES     : 2 selecoes x linkage {average, complete} x K in {3,4,5}        -> 12
  - DBSCAN    : 2 selecoes x eps {0.25, 0.28} x minPts {3, 5}                  ->  8

Selecao A -> subset_vulnerabilidade_eta2.csv (csv 1)
Selecao B -> subset_vulnerabilidade_spearman.csv (csv 2)
lst_media nao entra no agrupamento nem nas metricas.

Metricas: Cotovelo (soma dos quadrados das distancias ao centroide, WCSS), Silhueta,
Davies-Bouldin e ARI bootstrap. Ruido (-1, so DBSCAN) fica de fora das tres primeiras.

Saidas (pasta outputs/sprint3):
  <tecnica>/<dataset>/   metricas_cenarios.csv, curva_cotovelo, todos_cenarios_* (cada figura em .png e .pdf)
                         e, para cada cenario, <cenario>_cotovelo/silhueta/davies_bouldin/
                         ari_bootstrap/mapa_clusters (.png e .pdf)
  todos_os_cenarios/     <dataset>/ com todas as tecnicas de cada csv, e a raiz com csv 1 + csv 2

Uso:
    python clustering_sprint3.py                 # usa notebooks e códigos/sprint3/Datasets
    python clustering_sprint3.py caminho/dados   # outra pasta com os dois CSVs
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # execucao em lote: grava as figuras sem abrir janelas
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_samples
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURACAO
# ============================================================================
RANDOM_STATE = 42
N_BOOTSTRAP = 100          # reamostragens para a estabilidade (ARI)
FRAC_BOOTSTRAP = 0.8       # fracao dos bairros sorteada (sem reposicao) em cada reamostragem
CONTROLE = "lst_media"
COLS_NAO_VARIAVEIS = ["codbairro", "bairro", "regiao_administrativa",
                      "cluster_id", "cluster_nome", CONTROLE]
N_VARIAVEIS = 5

RAIZ = Path(__file__).resolve().parents[3]  # raiz do Drive (pasta que contem data/ e outputs/)
PASTA_DADOS_PADRAO = RAIZ / "notebooks e códigos" / "sprint3" / "Datasets"
PASTA_SAIDA = RAIZ / "outputs" / "sprint3"
if not PASTA_SAIDA.parent.is_dir():
    raise FileNotFoundError(f"RAIZ incorreta ({RAIZ}): pasta outputs/ nao encontrada. "
                            "Mantenha o script em notebooks e códigos/sprint3/<pasta>/.")
GPKG_BAIRROS = RAIZ / "outputs" / "socioeconomico" / "bairros_socioeconomico.gpkg"
ARQUIVOS = {
    "A": "subset_vulnerabilidade_eta2.csv",      # selecao A - relevancia (csv 1)
    "B": "subset_vulnerabilidade_spearman.csv",  # selecao B - redundancia (csv 2)
}
TEC_KMEDOIDS, TEC_AGNES, TEC_DBSCAN = "K-Medoids", "AGNES", "DBSCAN"
TECNICAS = [TEC_KMEDOIDS, TEC_AGNES, TEC_DBSCAN]

METRICAS = ["cotovelo", "sil", "db", "ari"]
ROTULOS_METRICAS = {
    "cotovelo": "Cotovelo (WCSS: soma dos quadrados intra-grupo, menor melhor)",
    "sil": "Silhueta (maior melhor)",
    "db": "Davies-Bouldin (menor melhor)",
    "ari": "ARI bootstrap (maior melhor)",
}
ARQUIVO_METRICA = {"cotovelo": "cotovelo", "sil": "silhueta",
                   "db": "davies_bouldin", "ari": "ari_bootstrap"}
EXTENSAO_FIGURA = ".pdf"  # formato das figuras para o LaTeX; usado por _salvar e pelo figuras.tex


# ============================================================================
# BLOCO 1 - DADOS
# ============================================================================
def carregar_dataset(caminho: Path) -> tuple[pd.DataFrame, list[str]]:
    """Le um CSV, ordena por codbairro e separa as variaveis de agrupamento."""
    df = pd.read_csv(caminho).sort_values("codbairro").reset_index(drop=True)
    features = [c for c in df.columns if c not in COLS_NAO_VARIAVEIS]
    if len(features) != N_VARIAVEIS:
        raise ValueError(f"{caminho.name}: esperadas {N_VARIAVEIS} variaveis, "
                         f"encontradas {len(features)} ({features})")
    return df, features


def normalizar(X: np.ndarray) -> np.ndarray:
    """Imputacao pela mediana e Min-Max sobre todos os 162 bairros."""
    imputado = SimpleImputer(strategy="median").fit_transform(X)
    return MinMaxScaler().fit_transform(imputado)


# ============================================================================
# BLOCO 2 - K-MEDOIDS (PAM: BUILD + SWAP, implementacao vetorizada)
# ============================================================================
def _pam(D: np.ndarray, k: int) -> np.ndarray:
    """Retorna os indices dos k medoides minimizando a soma das distancias ao medoide."""
    # BUILD: primeiro o ponto mais central; depois, o de maior ganho
    medoides = [int(np.argmin(D.sum(axis=1)))]
    while len(medoides) < k:
        dmin = D[:, medoides].min(axis=1)
        ganho = np.maximum(dmin[:, None] - D, 0).sum(axis=0)
        ganho[medoides] = -1
        medoides.append(int(np.argmax(ganho)))

    # SWAP: troca medoide i por candidato h enquanto houver melhora
    custo = D[:, medoides].min(axis=1).sum()
    while True:
        melhor = (custo, None, None)
        for i in range(k):
            resto = [m for j, m in enumerate(medoides) if j != i]
            d_resto = D[:, resto].min(axis=1)
            # custo para TODOS os candidatos h de uma vez (matriz n x n)
            custos = np.minimum(d_resto[:, None], D).sum(axis=0)
            custos[medoides] = np.inf
            h = int(np.argmin(custos))
            if custos[h] < melhor[0] - 1e-12:
                melhor = (custos[h], i, h)
        if melhor[1] is None:
            break
        custo, i, h = melhor
        medoides[i] = h
    return np.array(medoides)


class KMedoids:
    def __init__(self, k: int, metrica: str):
        self.k, self.metrica = k, metrica

    def fit(self, X: np.ndarray) -> "KMedoids":
        D = cdist(X, X, self.metrica)
        idx = _pam(D, self.k)
        self.labels_ = np.argmin(D[:, idx], axis=1)
        return self


# ============================================================================
# BLOCO 3 - AGNES (clustering hierarquico aglomerativo)
# ============================================================================
class AGNES:
    def __init__(self, k: int, linkage: str):
        self.k, self.linkage = k, linkage

    def fit(self, X: np.ndarray) -> "AGNES":
        self.labels_ = AgglomerativeClustering(
            n_clusters=self.k, metric="euclidean", linkage=self.linkage
        ).fit_predict(X)
        return self


# ============================================================================
# BLOCO 4 - DBSCAN
# ============================================================================
class DBSCANModel:
    def __init__(self, eps: float, min_pts: int):
        self.eps, self.min_pts = eps, min_pts

    def fit(self, X: np.ndarray) -> "DBSCANModel":
        self.labels_ = DBSCAN(eps=self.eps, min_samples=self.min_pts).fit(X).labels_
        return self


# ============================================================================
# Definicao dos 32 cenarios
# ============================================================================
@dataclass
class Cenario:
    id: int
    tecnica: str
    selecao: str
    params: dict = field(default_factory=dict)
    rotulo: str = ""
    config: str = ""
    serie: str = ""  # nome da curva no grafico do cotovelo

    def criar_modelo(self):
        p = self.params
        if self.tecnica == TEC_KMEDOIDS:
            return KMedoids(p["k"], p["metrica"])
        if self.tecnica == TEC_AGNES:
            return AGNES(p["k"], p["linkage"])
        if self.tecnica == TEC_DBSCAN:
            return DBSCANModel(p["eps"], p["min_pts"])
        raise ValueError(f"tecnica desconhecida: {self.tecnica!r}")


def gerar_cenarios() -> list[Cenario]:
    km = [dict(selecao=sel, params={"k": k, "metrica": metrica},
               rotulo=f"KM {sel} k{k} {nome[:3]}",
               config=f"Sel {sel}, K={k}, {nome}",
               serie=f"K-Medoids {sel} {nome}")
          for sel, k, (metrica, nome) in product(
              "AB", (3, 4, 5), (("cityblock", "Manhattan"), ("euclidean", "Euclidiana")))]
    ag = [dict(selecao=sel, params={"k": k, "linkage": linkage},
               rotulo=f"AGN {sel} {linkage[:3]} k{k}",
               config=f"Sel {sel}, linkage={linkage}, K={k}",
               serie=f"AGNES {sel} {linkage}")
          for sel, linkage, k in product("AB", ("average", "complete"), (3, 4, 5))]
    db = [dict(selecao=sel, params={"eps": eps, "min_pts": min_pts},
               rotulo=f"DBS {sel} e{eps} m{min_pts}",
               config=f"Sel {sel}, eps={eps}, minPts={min_pts}",
               serie=f"DBSCAN {sel}")
          for sel, eps, min_pts in product("AB", (0.25, 0.28), (3, 5))]

    cenarios: list[Cenario] = []
    for tecnica, lista in ((TEC_KMEDOIDS, km), (TEC_AGNES, ag), (TEC_DBSCAN, db)):
        for i, c in enumerate(lista, start=1):
            cenarios.append(Cenario(id=i, tecnica=tecnica, **c))
    return cenarios


# ============================================================================
# BLOCO 5 - METRICAS
# ============================================================================
def cotovelo_por_grupo(X: np.ndarray, rot: np.ndarray) -> dict:
    """Soma dos quadrados das distancias ao centroide, por grupo (ruido fora)."""
    wcss = {}
    for g in np.unique(rot[rot != -1]):
        Xg = X[rot == g]
        wcss[int(g)] = float(((Xg - Xg.mean(axis=0)) ** 2).sum())
    return wcss


def db_por_grupo(Xv: np.ndarray, rv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Contribuicao R_i de cada grupo no Davies-Bouldin (a media dos R_i e o indice)."""
    grupos = np.unique(rv)
    cent = np.vstack([Xv[rv == g].mean(axis=0) for g in grupos])
    s = np.array([np.linalg.norm(Xv[rv == g] - cent[i], axis=1).mean()
                  for i, g in enumerate(grupos)])
    M = cdist(cent, cent)
    np.fill_diagonal(M, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):  # centroides coincidentes -> inf
        R = np.max((s[:, None] + s[None, :]) / M, axis=1)
    return grupos, R


def metricas(X: np.ndarray, rot: np.ndarray) -> tuple[dict, dict]:
    """Retorna (metricas, dados_graficos). Cotovelo, silhueta e DB sobre os bairros fora do ruido."""
    valido = rot != -1
    k_efetivo = len(np.unique(rot[valido]))
    wcss = cotovelo_por_grupo(X, rot)
    met = {"cotovelo": sum(wcss.values()) if wcss else np.nan, "sil": np.nan, "db": np.nan}
    graf = {"wcss": wcss, "silhueta_amostras": None, "db_grupos": None, "db_R": None}
    if valido.sum() > k_efetivo >= 2:
        Xv, rv = X[valido], rot[valido]
        amostras = silhouette_samples(Xv, rv)
        grupos, R = db_por_grupo(Xv, rv)
        met.update(sil=float(amostras.mean()), db=davies_bouldin_score(Xv, rv))
        # ordena por grupo e, dentro do grupo, do maior para o menor valor (grafico de silhueta)
        amostras = amostras[np.lexsort((-amostras, rv))]
        graf.update(silhueta_amostras=amostras, db_grupos=grupos, db_R=R)
    return met, graf


def estabilidade_ari(cen: Cenario, X: np.ndarray, rot: np.ndarray, rng) -> np.ndarray:
    """ARI de cada reamostragem (80% dos bairros) contra o agrupamento completo.

    Ruido (-1) fica de fora do ARI. No DBSCAN, eps fixo em 80% dos bairros reduz a
    densidade, entao o ARI dele tende a ser mais baixo que o das outras tecnicas.
    """
    n = len(X)
    m = int(round(FRAC_BOOTSTRAP * n))
    ari = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(n, m, replace=False)
        ref, rb = rot[idx], cen.criar_modelo().fit(X[idx]).labels_
        ok = (ref != -1) & (rb != -1)  # ruido (-1) fora do ARI: nao e um grupo
        ari[b] = adjusted_rand_score(ref[ok], rb[ok]) if ok.sum() > 1 else np.nan
    return ari


# ============================================================================
# BLOCO 6 - GRAFICOS
# ============================================================================
def _salvar(fig, caminho: Path, dpi: int = 130) -> None:
    """Grava PNG (visualizacao) e PDF vetorial (incluir no LaTeX). bbox_inches evita titulos cortados."""
    fig.tight_layout()
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight")
    fig.savefig(caminho.with_suffix(EXTENSAO_FIGURA), bbox_inches="tight")
    plt.close(fig)


def _sem_dado(titulo: str, motivo: str, caminho: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, motivo, ha="center", va="center", fontsize=12)
    ax.set_axis_off()
    ax.set_title(titulo)
    _salvar(fig, caminho)


def plot_barras(rotulos, valores, erros, titulo, ylabel, caminho: Path, rot_x=90) -> None:
    """Uma barra por cenario. erros (desvio do bootstrap) e opcional."""
    x = np.arange(len(rotulos))
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(rotulos) + 3), 5.5))
    ax.bar(x, valores, yerr=erros, capsize=2, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, rotation=rot_x, fontsize=8)
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    _salvar(fig, caminho)


def plot_silhueta_cenario(titulo, sil_amostras, sil_media, caminho: Path) -> None:
    """Grafico de silhueta classico: uma barra por bairro, agrupadas por grupo."""
    if sil_amostras is None:
        _sem_dado(titulo, "Silhueta indisponivel (menos de 2 grupos)", caminho)
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(sil_amostras))
    ax.barh(y, sil_amostras, height=1.0, color="tab:blue")
    ax.axvline(sil_media, color="red", linestyle="--", label=f"media {sil_media:.3f}")
    ax.set_xlabel("Silhueta de cada bairro (maior melhor)")
    ax.set_ylabel("bairros agrupados por grupo")
    ax.set_yticks([])
    ax.set_title(titulo)
    ax.legend()
    _salvar(fig, caminho)


def plot_cotovelo_cenario(titulo, wcss: dict, caminho: Path) -> None:
    """WCSS de cada grupo (o total e a soma destas barras)."""
    if not wcss:
        _sem_dado(titulo, "Sem grupos (so ruido)", caminho)
        return
    grupos = list(wcss.keys())
    valores = [wcss[g] for g in grupos]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([f"grupo {g}" for g in grupos], valores, color="tab:blue")
    ax.set_ylabel("Soma dos quadrados das distancias ao centroide (WCSS)")
    ax.set_title(f"{titulo}\ntotal = {sum(valores):.3f}")
    ax.grid(axis="y", alpha=0.3)
    _salvar(fig, caminho)


def plot_db_cenario(titulo, grupos, R, db, caminho: Path) -> None:
    """Contribuicao R_i de cada grupo; a media dos R_i e o Davies-Bouldin."""
    if R is None:
        _sem_dado(titulo, "Davies-Bouldin indisponivel (menos de 2 grupos)", caminho)
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    # centroides coincidentes geram R_i infinito: esses grupos ficam sem barra
    ax.bar([f"grupo {g}" for g in grupos], np.where(np.isfinite(R), R, np.nan), color="tab:blue")
    if np.isfinite(db):
        ax.axhline(db, color="red", linestyle="--", label=f"DB = media {db:.3f}")
        ax.legend()
    ax.set_ylabel("R_i (pior par de cada grupo)")
    ax.set_title(titulo)
    ax.grid(axis="y", alpha=0.3)
    _salvar(fig, caminho)


def plot_hist_ari(titulo, ari: np.ndarray, caminho: Path) -> None:
    ari = ari[np.isfinite(ari)]  # reamostragens so com ruido dao NaN
    if ari.size == 0:
        _sem_dado(titulo, "ARI indisponivel (todas as reamostragens sem grupos)", caminho)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ari, bins=25, color="tab:blue", alpha=0.8)
    ax.axvline(np.nanmean(ari), color="red", linestyle="--",
               label=f"media {np.nanmean(ari):.3f} (dp {(ari.std(ddof=1) if ari.size > 1 else float('nan')):.3f})")
    ax.set_xlabel("ARI entre a reamostragem e o agrupamento completo (maior melhor)")
    ax.set_ylabel("Frequencia")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend()
    _salvar(fig, caminho)


def plot_curva_cotovelo(resumo: pd.DataFrame, caminho: Path) -> None:
    """Cotovelo: soma intra-grupo x K, uma curva por serie (so tecnicas com K)."""
    dados = resumo.assign(k=resumo["k"].astype(float))
    dados = dados[dados["k"].notna()]
    if dados.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for serie, sub in dados.groupby("serie"):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["cotovelo"], marker="o", label=serie)
    ax.set_xlabel("K")
    ax.set_ylabel("Soma dos quadrados das distancias ao centroide (WCSS)")
    ax.set_xticks([3, 4, 5])
    ax.set_title("Cotovelo: procurar a curva onde a queda de K para K+1 perde forca")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _salvar(fig, caminho)


def preparar_base_mapa() -> gpd.GeoDataFrame:
    base = gpd.read_file(GPKG_BAIRROS)[["codbairro", "geometry"]]
    base["codbairro"] = base["codbairro"].astype(int)
    return base.set_index("codbairro")  # indexado uma vez; cada mapa so faz o join


def desenhar_mapa(ax, base: gpd.GeoDataFrame, mapa_df: pd.DataFrame, titulo: str) -> None:
    """Mapa dos bairros colorido pelo grupo; bairros sem dado ficam cinza."""
    gdf = base.join(mapa_df.set_index("codbairro"), how="inner")  # so bairros com grupo
    if len(gdf) != len(mapa_df):
        print(f"AVISO: {len(mapa_df) - len(gdf)} bairro(s) sem poligono no mapa ({titulo})")
    grupos = ["ruido" if int(v) == -1 else f"grupo {int(v)}" for v in gdf["cluster"]]
    gdf = gdf.assign(grupo=grupos)
    gdf.plot(column="grupo", categorical=True, cmap="tab10", legend=True, ax=ax,
             edgecolor="white", linewidth=0.3)
    ax.set_title(titulo, fontsize=9)
    ax.set_axis_off()


def plot_grade_mapas(base, lista, caminho: Path, ncols: int = 4) -> None:
    """Grade com um mapa de clusters por cenario. lista = [(titulo, mapa_df), ...]."""
    n = len(lista)
    nrows = -(-n // ncols)
    fig, eixos = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    eixos = np.atleast_1d(eixos).ravel()
    for ax, (titulo, mapa_df) in zip(eixos, lista):
        desenhar_mapa(ax, base, mapa_df, titulo)
    for ax in eixos[n:]:
        ax.set_axis_off()
    _salvar(fig, caminho, dpi=110)


def gerar_saidas_cenario(c: Cenario, pasta: Path, det: dict, base) -> list[tuple[str, str]]:
    """Grava as figuras de um cenario. Retorna (nome sem extensao, legenda) de cada uma."""
    nome = c.rotulo.replace(" ", "_")
    plot_cotovelo_cenario(f"{c.rotulo} - cotovelo", det["wcss"], pasta / f"{nome}_cotovelo.png")
    plot_silhueta_cenario(f"{c.rotulo} - silhueta", det["silhueta_amostras"], det["sil"],
                          pasta / f"{nome}_silhueta.png")
    plot_db_cenario(f"{c.rotulo} - Davies-Bouldin", det["db_grupos"], det["db_R"], det["db"],
                    pasta / f"{nome}_davies_bouldin.png")
    plot_hist_ari(f"{c.rotulo} - ARI bootstrap ({N_BOOTSTRAP} reamostragens)",
                  det["ari"], pasta / f"{nome}_ari_bootstrap.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    desenhar_mapa(ax, base, det["mapa"], f"{c.rotulo} - clusters")
    _salvar(fig, pasta / f"{nome}_mapa_clusters.png")
    return [(f"{nome}_cotovelo", "Cotovelo"), (f"{nome}_silhueta", "Silhueta"),
            (f"{nome}_davies_bouldin", "Davies-Bouldin"), (f"{nome}_ari_bootstrap", "ARI bootstrap"),
            (f"{nome}_mapa_clusters", "Mapa de clusters")]


def gerar_saidas_combinadas(pasta: Path, resumo: pd.DataFrame, mapas_lista: list,
                            base, titulo_base: str) -> None:
    """Graficos com todos os cenarios juntos: uma barra por cenario para cada metrica."""
    rotulos = list(resumo["rotulo"])
    for m in METRICAS:
        valores = resumo["ari_mean"] if m == "ari" else resumo[m]
        erros = resumo["ari_std"] if m == "ari" else None  # so o ARI tem desvio do bootstrap
        plot_barras(rotulos, valores, erros,
                    f"{titulo_base} - {ROTULOS_METRICAS[m]}", ROTULOS_METRICAS[m],
                    pasta / f"todos_cenarios_{ARQUIVO_METRICA[m]}.png")
    plot_curva_cotovelo(resumo, pasta / "curva_cotovelo.png")
    plot_grade_mapas(base, [(c.rotulo, mapa) for c, mapa in mapas_lista],
                     pasta / "todos_cenarios_mapas_clusters.png")


# ============================================================================
# BLOCO 7 - EXECUCAO
# ============================================================================


def escapar_latex(texto: str) -> str:
    for antigo, novo in [("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"),
                         ("&", "\\&"), ("#", "\\#"), ("$", "\\$")]:
        texto = texto.replace(antigo, novo)
    return texto


def entradas_combinadas(com_curva: bool = True) -> list[tuple[str, str]]:
    """Figuras de todos os cenarios juntos (caminhos relativos a pasta). A curva so existe com K."""
    entradas = [(f"todos_cenarios_{ARQUIVO_METRICA[m]}{EXTENSAO_FIGURA}", ROTULOS_METRICAS[m]) for m in METRICAS]
    if com_curva:
        entradas.append((f"curva_cotovelo{EXTENSAO_FIGURA}", "Curva do cotovelo por K"))
    return entradas + [(f"todos_cenarios_mapas_clusters{EXTENSAO_FIGURA}",
                        "Mapas de clusters de todos os cenarios")]


def escrever_tex(pasta: Path, titulo: str, entradas: list[tuple[str, str]]) -> None:
    """figuras.tex: inclui os PDFs de uma pasta com \\includegraphics (compila com pdflatex)."""
    linhas = [r"\documentclass{article}", r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}",
              r"\usepackage{graphicx}", r"\usepackage{float}", r"\usepackage[margin=2cm]{geometry}",
              r"\begin{document}", rf"\section*{{{escapar_latex(titulo)}}}"]
    for rel, legenda in entradas:
        if (pasta / rel).exists():
            linhas += [r"\begin{figure}[H]", r"\centering", rf"\includegraphics[width=\linewidth]{{{rel}}}",
                       rf"\caption{{{escapar_latex(legenda)}}}", r"\end{figure}"]
        else:
            print(f"AVISO: figura ausente, nao incluida em figuras.tex: {pasta / rel}")
    linhas.append(r"\end{document}")
    (pasta / "figuras.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main() -> None:
    pasta_dados = Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA_DADOS_PADRAO
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    dados = {sel: carregar_dataset(pasta_dados / arq) for sel, arq in ARQUIVOS.items()}
    (df_a, _), (df_b, _) = dados["A"], dados["B"]
    if not np.array_equal(df_a["codbairro"].values, df_b["codbairro"].values):
        raise ValueError("Os dois datasets nao tem os mesmos bairros (codbairro).")
    base = preparar_base_mapa()
    codbairros = df_a["codbairro"].to_numpy()
    base = base[base.index.isin(codbairros)]  # so os 162 bairros do dataset entram no mapa

    X_por_sel = {sel: normalizar(df[feats].to_numpy(float)) for sel, (df, feats) in dados.items()}

    cenarios = gerar_cenarios()
    linhas = []
    detalhes: dict[tuple, dict] = {}
    for c in cenarios:
        X = X_por_sel[c.selecao]
        rot = c.criar_modelo().fit(X).labels_
        met, graf = metricas(X, rot)
        # RNG por cenario: independente da ordem da lista de cenarios
        rng = np.random.default_rng([RANDOM_STATE, TECNICAS.index(c.tecnica), ord(c.selecao), c.id])
        ari = estabilidade_ari(c, X, rot, rng)
        det = {**met, **graf, "ari": ari,
               "mapa": pd.DataFrame({"codbairro": codbairros, "cluster": rot})}
        detalhes[(c.tecnica, c.selecao, c.id)] = det
        linhas.append({"id": c.id, "tecnica": c.tecnica, "selecao": c.selecao,
                       "serie": c.serie, "k": c.params.get("k"), "rotulo": c.rotulo,
                       "config": c.config, "cotovelo": met["cotovelo"], "sil": met["sil"],
                       "db": met["db"], "ari_mean": np.nanmean(ari), "ari_std": np.nanstd(ari, ddof=1)})
        print(f"{c.rotulo:<14} concluido")

    resumo = (pd.DataFrame(linhas)
              .sort_values(["tecnica", "selecao", "id"]).reset_index(drop=True))

    for tec in TECNICAS:
        for sel, arq in ARQUIVOS.items():
            pasta = PASTA_SAIDA / tec / Path(arq).stem
            pasta.mkdir(parents=True, exist_ok=True)
            sub = resumo[(resumo["tecnica"] == tec) & (resumo["selecao"] == sel)]
            sub.to_csv(pasta / "metricas_cenarios.csv", index=False, float_format="%.4f")
            mapas_lista = []
            entradas = []
            for c in (c for c in cenarios if c.tecnica == tec and c.selecao == sel):
                det = detalhes[(tec, sel, c.id)]
                # cenarios com K ficam em k3/, k4/ ou k5/; DBSCAN (sem K) fica na pasta do dataset
                pasta_c = pasta / f"k{c.params['k']}" if "k" in c.params else pasta
                pasta_c.mkdir(exist_ok=True)
                gerados = gerar_saidas_cenario(c, pasta_c, det, base)
                mapas_lista.append((c, det["mapa"]))
                # caminhos saem da propria funcao que grava as figuras (sem recalcular nomes)
                rel_pasta = pasta_c.relative_to(pasta)
                entradas += [((rel_pasta / f"{stem}{EXTENSAO_FIGURA}").as_posix(), f"{c.rotulo} - {legenda}")
                             for stem, legenda in gerados]
            gerar_saidas_combinadas(pasta, sub, mapas_lista, base, f"{tec} - {arq}")
            escrever_tex(pasta, f"{tec} - {arq}", entradas + entradas_combinadas(tec != TEC_DBSCAN))
            print(f"Saidas de {tec} / {arq} gravadas em {pasta}")

    for sel, arq in ARQUIVOS.items():
        pasta_sel = PASTA_SAIDA / "todos_os_cenarios" / Path(arq).stem
        pasta_sel.mkdir(parents=True, exist_ok=True)
        sub = resumo[resumo["selecao"] == sel]
        sub.to_csv(pasta_sel / "metricas_todas_tecnicas.csv", index=False, float_format="%.4f")
        mapas_sel = [(c, detalhes[(c.tecnica, c.selecao, c.id)]["mapa"])
                     for c in cenarios if c.selecao == sel]
        gerar_saidas_combinadas(pasta_sel, sub, mapas_sel, base, f"Todas as tecnicas - {arq}")
        escrever_tex(pasta_sel, f"Todas as tecnicas - {arq}", entradas_combinadas())
        print(f"Saidas de todas as tecnicas / {arq} gravadas em {pasta_sel}")

    pasta_todos = PASTA_SAIDA / "todos_os_cenarios"
    pasta_todos.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(pasta_todos / "metricas_todos_cenarios.csv", index=False, float_format="%.4f")
    mapas_todos = [(c, detalhes[(c.tecnica, c.selecao, c.id)]["mapa"]) for c in cenarios]
    gerar_saidas_combinadas(pasta_todos, resumo, mapas_todos, base, "Todos os cenarios")
    escrever_tex(pasta_todos, "Todos os cenarios (csv 1 + csv 2)", entradas_combinadas())

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== Metricas por cenario (ajuste nos 162 bairros) ===")
    print(resumo[["rotulo", "config", "cotovelo", "sil", "db", "ari_mean", "ari_std"]]
          .round(3).to_string(index=False))
    print(f"\nResultados salvos em: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()
