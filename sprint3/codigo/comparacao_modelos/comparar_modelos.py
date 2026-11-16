"""Comparacao unificada de todos os cenarios de clustering (K-Medoids, AGNES, DBSCAN).

Le metricas_todos_cenarios.csv (cada cenario ajustado nos 162 bairros, sem folds) e
acrescenta o numero de grupos e a % de ruido (essenciais para o DBSCAN,
cuja silhueta/DB nao existe quando ha menos de 2 grupos). Gera:

  comparacao_modelos/tabela_comparativa.csv   uma linha por cenario, todas as metricas
  comparacao_modelos/ranking_geral.csv        ranking composto (so cenarios validos)
  comparacao_modelos/melhor_por_tecnica.csv   melhor cenario de cada tecnica x selecao
  comparacao_modelos/*.png                    graficos comparativos

Criterios do ranking: silhueta (maior), Davies-Bouldin (menor) e ARI bootstrap (maior).
  Cada criterio vira posicao percentual entre os cenarios validos e o score e a media
  das posicoes. O cotovelo fica fora: a soma intra-grupo cai com K por
  construcao e nao e comparavel entre tecnicas nem entre Ks.
Cenario valido: >= 2 grupos e <= RUIDO_MAX de ruido na base completa.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codigo_clustering_kmedoids"))
import clustering_sprint3 as cs  # noqa: E402

RUIDO_MAX = 0.20
PASTA = cs.PASTA_SAIDA / "comparacao_modelos"
CORES = {"K-Medoids": "#1f77b4", "AGNES": "#2ca02c", "DBSCAN": "#d62728"}


def perfil_base_completa() -> pd.DataFrame:
    """Ajusta cada cenario em todos os bairros: n de grupos, % de ruido, silhueta/DB."""
    dados = {s: cs.carregar_dataset(cs.PASTA_DADOS_PADRAO / arq) for s, arq in cs.ARQUIVOS.items()}
    resumo_csv = cs.PASTA_SAIDA / "todos_os_cenarios" / "metricas_todos_cenarios.csv"  # noqa: F841
    linhas = []
    for cen in cs.gerar_cenarios():
        df, feats = dados[cen.selecao]
        X = cs.normalizar(df[feats].to_numpy(float))
        rot = cen.criar_modelo().fit(X).labels_
        valido = rot != -1
        grupos = np.unique(rot[valido])
        tamanhos = np.array([(rot == g).sum() for g in grupos])
        m, _ = cs.metricas(X, rot)
        linhas.append({
            "rotulo": cen.rotulo, "n_grupos": len(grupos),
            "pct_ruido": 100 * (~valido).mean(),
            "maior_grupo_pct": 100 * tamanhos.max() / len(rot) if len(grupos) else np.nan,
            "menor_grupo_n": int(tamanhos.min()) if len(grupos) else 0,
        })
    return pd.DataFrame(linhas)


def pos_percentual(s: pd.Series, maior_melhor: bool) -> pd.Series:
    return s.rank(pct=True, ascending=maior_melhor, na_option="keep")


def montar_tabelas() -> tuple[pd.DataFrame, pd.DataFrame]:
    res = pd.read_csv(cs.PASTA_SAIDA / "todos_os_cenarios" / "metricas_todos_cenarios.csv")
    tab = res.merge(perfil_base_completa(), on="rotulo", how="left")
    tab["valido"] = (tab["n_grupos"] >= 2) & (tab["pct_ruido"] <= 100 * RUIDO_MAX)
    motivo = np.where(tab["n_grupos"] < 2, "menos de 2 grupos",
                      np.where(tab["pct_ruido"] > 100 * RUIDO_MAX, f"ruido > {RUIDO_MAX:.0%}", ""))
    tab["motivo_exclusao"] = motivo

    v = tab[tab["valido"]].copy()
    v["p_sil"] = pos_percentual(v["sil"], True)
    v["p_db"] = pos_percentual(v["db"], False)
    v["p_ari"] = pos_percentual(v["ari_mean"], True)
    v["score"] = v[["p_sil", "p_db", "p_ari"]].mean(axis=1)
    v = v.sort_values("score", ascending=False).reset_index(drop=True)
    v.insert(0, "posicao", v.index + 1)
    tab = tab.merge(v[["rotulo", "posicao", "score"]], on="rotulo", how="left")
    return tab.sort_values(["posicao", "tecnica"]), v


def graficos(tab: pd.DataFrame, rank: pd.DataFrame) -> None:
    # 1) qualidade x estabilidade, bolha = % ruido
    fig, ax = plt.subplots(figsize=(10, 7))
    for tec, sub in tab.groupby("tecnica"):
        v = sub[sub["valido"]]
        ax.scatter(v["sil"], v["ari_mean"], s=60 + 12 * v["pct_ruido"],
                   c=CORES[tec], alpha=0.7, label=tec, edgecolor="k")
        for _, r in v.iterrows():
            ax.annotate(r["rotulo"], (r["sil"], r["ari_mean"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Silhueta (maior melhor)")
    ax.set_ylabel("ARI bootstrap (maior melhor)")
    ax.set_title("Qualidade x estabilidade (so cenarios validos; tamanho = % ruido)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PASTA / "qualidade_x_estabilidade.png", dpi=130)
    plt.close(fig)

    # 2) ranking composto
    top = rank.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["rotulo"], top["score"], color=[CORES[t] for t in top["tecnica"]])
    ax.set_xlabel("Score composto (0-1, maior melhor)")
    ax.set_title("Top 20 cenarios validos")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA / "ranking_geral.png", dpi=130)
    plt.close(fig)

    # 3) distribuicao por tecnica de cada metrica
    metr = [("sil", "Silhueta (maior melhor)"), ("db", "Davies-Bouldin (menor melhor)"),
            ("ari_mean", "ARI (maior melhor)")]
    fig, eixos = plt.subplots(1, 3, figsize=(14, 5))
    v = tab[tab["valido"]]
    tecs = list(CORES)
    for ax, (col, titulo) in zip(eixos, metr):
        dados = [v.loc[v["tecnica"] == t, col].dropna() for t in tecs]
        ax.boxplot(dados)
        ax.set_xticks(range(1, len(tecs) + 1))
        ax.set_xticklabels(tecs)
        for i, d in enumerate(dados, 1):
            ax.scatter(np.full(len(d), i), d, c=CORES[tecs[i - 1]], zorder=3, s=20)
        ax.set_title(titulo, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Metricas por tecnica (cenarios validos)")
    fig.tight_layout()
    fig.savefig(PASTA / "metricas_por_tecnica.png", dpi=130)
    plt.close(fig)

    # 4) numero de grupos e ruido de todos os cenarios (inclui os excluidos)
    t = tab.sort_values(["tecnica", "selecao", "id"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(len(t))
    cor = [CORES[c] for c in t["tecnica"]]
    a1.bar(x, t["n_grupos"], color=cor)
    a1.set_ylabel("N de grupos (base completa)")
    a2.bar(x, t["pct_ruido"], color=cor)
    a2.axhline(100 * RUIDO_MAX, ls="--", c="k")
    a2.set_ylabel("% ruido")
    a2.set_xticks(x)
    a2.set_xticklabels(t["rotulo"], rotation=90, fontsize=7)
    fig.suptitle("Estrutura de cada cenario (linha tracejada = limite de ruido)")
    fig.tight_layout()
    fig.savefig(PASTA / "grupos_e_ruido.png", dpi=130)
    plt.close(fig)


def main() -> None:
    PASTA.mkdir(parents=True, exist_ok=True)
    tab, rank = montar_tabelas()
    tab.to_csv(PASTA / "tabela_comparativa.csv", index=False, float_format="%.4f")
    rank.to_csv(PASTA / "ranking_geral.csv", index=False, float_format="%.4f")
    melhor = rank.sort_values("score", ascending=False).groupby(["tecnica", "selecao"]).head(1)
    melhor.to_csv(PASTA / "melhor_por_tecnica.csv", index=False, float_format="%.4f")
    graficos(tab, rank)

    cols = ["posicao", "rotulo", "n_grupos", "pct_ruido", "sil", "db", "ari_mean", "score"]
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("=== Top 10 ===")
    print(rank[cols].head(10).round(3).to_string(index=False))
    print("\n=== Melhor de cada tecnica x selecao ===")
    print(melhor[cols].round(3).to_string(index=False))
    exc = tab[~tab["valido"]][["rotulo", "n_grupos", "pct_ruido", "motivo_exclusao"]]
    print(f"\n=== Excluidos do ranking ({len(exc)}) ===")
    print(exc.round(1).to_string(index=False))
    print(f"\nSaidas em: {PASTA}")


if __name__ == "__main__":
    main()
