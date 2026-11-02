"""
GERAÇÃO DO MAPA INTERATIVO HTML: MODELO K-MEDOIDS V2 (VULNERABILIDADE SOCIOAMBIENTAL)
Ciclo de Ciência de Dados - Visualização Espacial Interativa dos 162 Bairros do Rio de Janeiro

Objetivo:
- Código 100% modularizado que permite ao programador alterar o valor de K:
    1. No Python: chamando 'gerar_mapa_html_kmedoids(k=5)' ou via terminal.
    2. Na interface HTML: através de um Seletor Interativo de K (K=2, 3, 4, 5, 6)
       que recalcula as cores, a legenda e os medoides arquetípicos em tempo real sem recarregar a página!
- Camada de Confirmação Visual da Temperatura de Superfície (LST Real):
    * Permite alternar entre os Clusters de Vulnerabilidade Socioambiental e o mapa térmico
      de calor orbital real (LST), possibilitando validar visualmente a sobreposição entre as ilhas
      de calor físicas e os territórios de maior precariedade social.
- Modo Matriz 2D de Risco Integrado:
    * Exibe os 4/5 quadrantes de prioridade pública (Urgente, Alerta Físico, Acompanhamento, Preservação).
- Legenda dinâmica com contagem exata de bairros em cada cluster e medoides destacados.
- Marcadores arquetípicos visuais nos bairros medoides reais da cidade.
- Reprojeção de coordenadas UTM 23S para WGS84 e renderização de alta performance com Leaflet.js.
- Salva em 'outputs/mapa_ilhas_calor_kmedoids.html'.
"""

import json
import math
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import importlib


def utm23s_para_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """Converte coordenadas métricas UTM Zona 23S (EPSG:31983) em graus WGS84 (Lon, Lat)."""
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a**2 - b**2) / a**2
    e_prime2 = (a**2 - b**2) / b**2
    k0 = 0.9996
    lon0 = math.radians(-45.0)

    x = easting - 500000.0
    y = northing - 10000000.0

    M = y / k0
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    mu = M / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))

    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
        + (1097 * e1**4 / 512) * math.sin(8 * mu)
    )

    N1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    T1 = math.tan(phi1) ** 2
    C1 = e_prime2 * math.cos(phi1) ** 2
    R1 = a * (1 - e2) / ((1 - e2 * math.sin(phi1) ** 2) ** 1.5)
    D = x / (N1 * k0)

    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
        D**2 / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1**2 - 9 * e_prime2) * D**4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1**2 - 252 * e_prime2 - 3 * C1**2) * D**6 / 720
    )

    lon = lon0 + (
        D
        - (1 + 2 * T1 + C1) * D**3 / 6
        + (5 - 2 * C1 + 28 * T1 - 3 * C1**2 + 8 * e_prime2 + 24 * T1**2) * D**5 / 120
    ) / math.cos(phi1)

    return round(math.degrees(lon), 6), round(math.degrees(lat), 6)


def reprojetar_geometria(coords):
    """Reprojeta recursivamente anéis de polígonos ou multipolígonos para WGS84 [lon, lat]."""
    if isinstance(coords[0], (int, float)):
        return list(utm23s_para_wgs84(coords[0], coords[1]))
    return [reprojetar_geometria(c) for c in coords]


def gerar_paleta_vulnerabilidade(k: int) -> list[str]:
    """Gera uma paleta cromática perceptualmente coerente para vulnerabilidade (0=Verde até k-1=Vermelho)."""
    paletas = {
        2: ["#16a34a", "#dc2626"],
        3: ["#16a34a", "#f59e0b", "#dc2626"],
        4: ["#16a34a", "#0284c7", "#f59e0b", "#dc2626"],
        5: ["#16a34a", "#10b981", "#0284c7", "#f59e0b", "#dc2626"],
        6: ["#16a34a", "#10b981", "#06b6d4", "#0284c7", "#f59e0b", "#dc2626"],
    }
    return paletas.get(k, ["#16a34a", "#0284c7", "#f59e0b", "#dc2626"])


def obter_cor_lst(lst: float) -> tuple[str, str]:
    """Retorna cor e classe térmica para a temperatura de superfície observada (LST)."""
    if lst < 34.0:
        return "#2ca02c", "< 34.0°C (Refúgio Térmico)"
    elif lst < 37.0:
        return "#85e085", "34.0 - 37.0°C (Ameno)"
    elif lst < 39.5:
        return "#fed976", "37.0 - 39.5°C (Moderado)"
    elif lst < 41.5:
        return "#fd8d3c", "39.5 - 41.5°C (Calor Elevado)"
    elif lst < 43.5:
        return "#f03b20", "41.5 - 43.5°C (Ilha de Calor)"
    else:
        return "#bd0026", "> 43.5°C (Calor Extremo)"


def obter_cor_prioridade(prioridade: str) -> str:
    """Retorna a cor associada ao nível de prioridade na Matriz 2D."""
    if "Urgente" in prioridade:
        return "#dc2626"
    elif "Alerta" in prioridade:
        return "#ea580c"
    elif "Acompanhamento Prioritário" in prioridade:
        return "#eab308"
    elif "Acompanhamento" in prioridade:
        return "#2563eb"
    else:
        return "#16a34a"


def gerar_mapa_html_kmedoids(k_ativo: int = 4) -> Path:
    diretorio_atual = Path(__file__).resolve().parent
    diretorio_raiz = diretorio_atual
    while diretorio_raiz != diretorio_raiz.parent:
        if (diretorio_raiz / 'data').exists() or (diretorio_raiz / 'output').exists():
            break
        diretorio_raiz = diretorio_raiz.parent

    caminho_base_prep = diretorio_atual / "outputs" / "dados_preparados_kmedoids.csv"
    caminho_base_perfilada = diretorio_atual / "outputs" / "dados_perfilados_matriz_kmedoids.csv"

    candidatos_geo = [
        diretorio_atual / "dados" / "ivs_socioeconomico_bairros_rio.geojson",
        diretorio_atual / "ivs_socioeconomico_bairros_rio.geojson",
        diretorio_raiz / "data" / "processed" / "02_trilha_socioeconomica" / "ivs_socioeconomico_bairros_rio.geojson",
    ]
    caminho_geojson_orig = next((p for p in candidatos_geo if p.exists()), candidatos_geo[0])
    caminho_saida_html = diretorio_atual / "outputs" / "mapa_ilhas_calor_kmedoids.html"

    # Garantir pré-requisitos
    if not caminho_base_prep.exists():
        modulo_prep = importlib.import_module("03_preprocessamento_kmedoids")
        modulo_prep.executar_preprocessamento_kmedoids()

    if not caminho_base_perfilada.exists():
        modulo_perfil = importlib.import_module("05_perfilamento_matriz_kmedoids")
        modulo_perfil.executar_perfilamento_matriz_kmedoids()

    df_base = pd.read_csv(caminho_base_perfilada).sort_values("codbairro").reset_index(drop=True)
    df_prep = pd.read_csv(caminho_base_prep).sort_values("codbairro").reset_index(drop=True)

    print("=" * 70)
    print(f"GERAÇÃO DO MAPA INTERATIVO HTML V2 (VULNERABILIDADE & CONFIRMAÇÃO LST - K={k_ativo})")
    print("=" * 70)

    # 1. Pré-computação das soluções PAM para K in [2, 6]
    modulo_04 = importlib.import_module("04_modelagem_kmedoids")
    features_norm = [
        "renda_resp_mediana_norm",
        "pct_pop_favela_norm",
        "densidade_ajustada_norm",
        "arborizacao_integrada_norm",
        "pct_sem_saneamento_norm",
        "pct_pavimentacao_norm",
        "taxa_analfabetismo_norm",
    ]
    X_norm = df_prep[features_norm].values
    D_mat = modulo_04.calcular_matriz_manhattan(X_norm)

    solucoes_k = {}
    config_clusters_por_k = {}

    for k_val in range(2, 7):
        meds, assign, _ = modulo_04.pam_build_and_swap(D_mat, k_val)

        # Ordenar clusters por vulnerabilidade média crescente
        df_temp = pd.DataFrame({
            "assign": assign,
            "vuln": df_base["score_vulnerabilidade_100"],
            "lst": df_base["lst_media"],
        })
        medias_vuln = df_temp.groupby("assign")["vuln"].mean().sort_values()
        mapping = {old: new for new, old in enumerate(medias_vuln.index)}

        ord_assign = df_temp["assign"].map(mapping).values
        ord_meds = [meds[old] for old in medias_vuln.index]
        med_names = [df_base.iloc[m]["bairro"] for m in ord_meds]

        cores = gerar_paleta_vulnerabilidade(k_val)

        # Configuração semântica
        clusters_info = {}
        contagens = pd.Series(ord_assign).value_counts().to_dict()

        for c_id in range(k_val):
            nome_med = med_names[c_id]
            qtd = contagens.get(c_id, 0)
            if c_id == 0:
                nome_sem = f"Alta Resiliência / Baixa Vulnerabilidade (Arquétipo: {nome_med})"
            elif c_id == k_val - 1:
                nome_sem = f"Hiper-Vulnerabilidade Crítica (Arquétipo: {nome_med})"
            elif c_id == 1 and k_val >= 4:
                nome_sem = f"Vulnerabilidade Baixa-Média / Ilha de Calor (Arquétipo: {nome_med})"
            elif c_id == k_val - 2 and k_val >= 4:
                nome_sem = f"Vulnerabilidade Estrutural Intermediária (Arquétipo: {nome_med})"
            else:
                nome_sem = f"Vulnerabilidade Transição Nível {c_id} (Arquétipo: {nome_med})"

            clusters_info[c_id] = {
                "nome": nome_sem,
                "medoide": nome_med,
                "cor": cores[c_id],
                "qtd": int(qtd),
                "vuln_media": round(float(medias_vuln.iloc[c_id]), 1),
            }

        config_clusters_por_k[k_val] = clusters_info
        solucoes_k[k_val] = {
            "labels": ord_assign.tolist(),
            "medoids_indices": ord_meds,
            "medoids_names": med_names,
        }

    # 2. Dicionário enriquecido por bairro com soluções multi-K e dados de confirmação visual
    bairros_dict = df_base.set_index("codbairro").to_dict(orient="index")

    for idx_bairro, (cod, dados) in enumerate(bairros_dict.items()):
        # Adicionar cores e classes da confirmação térmica (LST)
        cor_lst, classe_lst = obter_cor_lst(float(dados["lst_media"]))
        dados["cor_lst"] = cor_lst
        dados["classe_lst"] = classe_lst
        dados["cor_prioridade"] = obter_cor_prioridade(str(dados["nivel_prioridade"]))

        # Injetar partições multi-K
        dados["clusters_por_k"] = {}
        for k_val in range(2, 7):
            cid = solucoes_k[k_val]["labels"][idx_bairro]
            c_info = config_clusters_por_k[k_val][cid]
            is_med = idx_bairro in solucoes_k[k_val]["medoids_indices"]
            dados["clusters_por_k"][str(k_val)] = {
                "cluster_id": cid,
                "cluster_nome": c_info["nome"],
                "medoide": c_info["medoide"],
                "is_medoide": is_med,
                "cor": c_info["cor"],
            }

    # 3. Reprojetar GeoJSON para WGS84
    with open(caminho_geojson_orig, "r", encoding="utf-8") as f:
        geo_orig = json.load(f)

    features_reprojetadas = []
    for feat in geo_orig["features"]:
        cod = int(feat["properties"].get("codbairro", -1))
        if cod in bairros_dict:
            dados = bairros_dict[cod]
            features_reprojetadas.append({
                "type": "Feature",
                "properties": dados,
                "geometry": {
                    "type": feat["geometry"]["type"],
                    "coordinates": reprojetar_geometria(feat["geometry"]["coordinates"]),
                },
            })

    geojson_final = {
        "type": "FeatureCollection",
        "features": features_reprojetadas,
    }

    # 4. Geração do Documento HTML Interativo com Modos de Visualização
    conteudo_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mapeamento Interativo de Vulnerabilidade & Ilhas de Calor (K-Medoids v2) - Rio de Janeiro</title>
  
  <!-- Leaflet CSS & JS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    body {{
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
      background: #0f172a;
      color: #1e293b;
    }}
    header {{
      background: #0f172a;
      color: #fff;
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #1e293b;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
      z-index: 1000;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .header-left {{
      display: flex;
      flex-direction: column;
    }}
    .header-title {{
      font-size: 1.15rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #f8fafc;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .header-sub {{
      font-size: 0.78rem;
      color: #94a3b8;
    }}
    .controls-group {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .mode-selector-container {{
      display: flex;
      align-items: center;
      gap: 4px;
      background: #1e293b;
      padding: 4px 6px;
      border-radius: 8px;
      border: 1px solid #334155;
    }}
    .mode-btn {{
      background: transparent;
      border: none;
      color: #94a3b8;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .mode-btn:hover {{
      color: #fff;
      background: #334155;
    }}
    .mode-btn.active {{
      background: #0284c7;
      color: #fff;
      box-shadow: 0 0 8px rgba(2, 132, 199, 0.5);
    }}
    .k-selector-container {{
      display: flex;
      align-items: center;
      gap: 6px;
      background: #1e293b;
      padding: 4px 8px;
      border-radius: 8px;
      border: 1px solid #334155;
      transition: opacity 0.2s ease;
    }}
    .k-selector-label {{
      font-size: 0.75rem;
      font-weight: 700;
      color: #38bdf8;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .k-btn {{
      background: #0f172a;
      border: 1px solid #475569;
      color: #cbd5e1;
      padding: 4px 10px;
      border-radius: 5px;
      font-size: 0.78rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .k-btn:hover {{
      background: #334155;
      color: #fff;
    }}
    .k-btn.active {{
      background: #38bdf8;
      border-color: #38bdf8;
      color: #0f172a;
      font-weight: 800;
    }}
    .cluster-filter-container {{
      display: flex;
      align-items: center;
      gap: 6px;
      background: #1e293b;
      padding: 4px 8px;
      border-radius: 8px;
      border: 1px solid #334155;
      transition: opacity 0.2s ease;
    }}
    #select-cluster-filter {{
      background: #0f172a;
      color: #f8fafc;
      border: 1px solid #475569;
      border-radius: 5px;
      padding: 4px 8px;
      font-size: 0.78rem;
      font-weight: 700;
      outline: none;
      cursor: pointer;
    }}
    #select-cluster-filter:focus {{
      border-color: #38bdf8;
    }}
    .btn-clear-cluster {{
      background: #ef4444;
      color: #fff;
      border: none;
      border-radius: 4px;
      padding: 3px 7px;
      font-size: 0.72rem;
      font-weight: 800;
      cursor: pointer;
      line-height: 1.2;
      transition: background 0.15s ease;
    }}
    .btn-clear-cluster:hover {{
      background: #dc2626;
    }}
    .search-box {{
      position: relative;
      width: 220px;
    }}
    .search-box input {{
      width: 100%;
      padding: 6px 12px;
      border-radius: 6px;
      border: 1px solid #334155;
      background: #1e293b;
      color: #fff;
      font-size: 0.82rem;
      outline: none;
    }}
    .search-box input:focus {{
      border-color: #38bdf8;
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }}
    #main-container {{
      display: flex;
      flex: 1;
      position: relative;
      overflow: hidden;
    }}
    #map {{
      flex: 1;
      height: 100%;
      background: #0b1120;
    }}
    #sidebar {{
      width: 410px;
      background: #ffffff;
      border-left: 1px solid #e2e8f0;
      display: flex;
      flex-direction: column;
      box-shadow: -4px 0 15px rgba(0, 0, 0, 0.1);
      z-index: 999;
      overflow-y: auto;
    }}
    .sidebar-header {{
      padding: 18px 20px;
      background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
    }}
    .bairro-nome {{
      font-size: 1.35rem;
      font-weight: 800;
      color: #0f172a;
      margin-bottom: 2px;
    }}
    .ra-nome {{
      font-size: 0.82rem;
      color: #64748b;
      margin-bottom: 10px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .badge-medoide {{
      background: #fef3c7;
      color: #b45309;
      border: 1px solid #fde68a;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 700;
    }}
    .sidebar-body {{
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }}
    .metric-card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 9px 12px;
    }}
    .metric-card.destaque-calor {{
      background: #fff1f2;
      border-color: #fecdd3;
    }}
    .metric-card.destaque-vuln {{
      background: #fff7ed;
      border-color: #fed7aa;
    }}
    .metric-label {{
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      color: #64748b;
      margin-bottom: 2px;
    }}
    .metric-value {{
      font-size: 1.15rem;
      font-weight: 800;
      color: #0f172a;
    }}
    .metric-card.destaque-calor .metric-value {{
      color: #e11d48;
    }}
    .metric-card.destaque-vuln .metric-value {{
      color: #ea580c;
    }}
    .features-list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px;
    }}
    .features-title {{
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
      color: #475569;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }}
    .feat-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.78rem;
    }}
    .feat-name {{
      color: #475569;
      font-weight: 500;
    }}
    .feat-val {{
      font-weight: 700;
      color: #0f172a;
    }}
    .progress-bar-bg {{
      width: 70px;
      height: 6px;
      background: #e2e8f0;
      border-radius: 3px;
      overflow: hidden;
      margin-left: 8px;
    }}
    .progress-bar-fill {{
      height: 100%;
      border-radius: 3px;
    }}
    .info-section {{
      background: #f1f5f9;
      border-radius: 8px;
      padding: 11px 13px;
      border-left: 4px solid #0284c7;
    }}
    .info-section h3 {{
      font-size: 0.74rem;
      text-transform: uppercase;
      color: #475569;
      font-weight: 800;
      margin-bottom: 3px;
      letter-spacing: 0.05em;
    }}
    .info-section p {{
      font-size: 0.82rem;
      color: #1e293b;
      line-height: 1.35;
    }}
    .maup-alert {{
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-left: 4px solid #ef4444;
      padding: 9px 11px;
      border-radius: 6px;
      font-size: 0.78rem;
      color: #991b1b;
      line-height: 1.35;
      font-weight: 500;
    }}
    .legend-control {{
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(8px);
      padding: 12px 14px;
      border-radius: 8px;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
      border: 1px solid #cbd5e1;
      max-width: 330px;
      font-size: 0.78rem;
      line-height: 1.35;
    }}
    .legend-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      padding-bottom: 6px;
      border-bottom: 1px solid #e2e8f0;
      font-weight: 800;
      color: #0f172a;
      font-size: 0.82rem;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      margin-bottom: 4px;
      gap: 8px;
      padding: 3px 6px;
      border-radius: 5px;
      cursor: pointer;
      transition: all 0.15s ease;
      border: 1px solid transparent;
    }}
    .legend-item:hover {{
      background: #f1f5f9;
      border-color: #cbd5e1;
    }}
    .legend-item.item-ativo {{
      background: #e0f2fe;
      border-color: #0284c7;
      font-weight: 700;
    }}
    .legend-item.item-apagado {{
      opacity: 0.45;
    }}
    .btn-legend-clear {{
      background: #ef4444;
      color: #ffffff;
      border: none;
      border-radius: 4px;
      padding: 2px 6px;
      font-size: 0.70rem;
      font-weight: 700;
      cursor: pointer;
    }}
    .btn-legend-clear:hover {{
      background: #dc2626;
    }}
    .legend-color {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid rgba(0,0,0,0.2);
      flex-shrink: 0;
    }}
    .legend-label {{
      font-size: 0.75rem;
      color: #334155;
      line-height: 1.25;
    }}
    .legend-count {{
      font-weight: 800;
      color: #0284c7;
      margin-left: 2px;
    }}
    .empty-state {{
      padding: 40px 20px;
      text-align: center;
      color: #64748b;
    }}
    .empty-state svg {{
      width: 48px;
      height: 48px;
      color: #94a3b8;
      margin-bottom: 12px;
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-left">
      <div class="header-title">
        <span>👥 Vulnerabilidade Socioespacial & Ilhas de Calor (K-Medoids v2)</span>
      </div>
      <div class="header-sub">Análise Multidimensional de 7 Vulnerabilidades + Confirmação Visual Térmica (LST)</div>
    </div>
    
    <div class="controls-group">
      <!-- Seletor de Modo de Visualização -->
      <div class="mode-selector-container">
        <button class="mode-btn active" id="mode-kmedoids" onclick="trocarModo('kmedoids')">
          👥 Clusters K-Medoids
        </button>
        <button class="mode-btn" id="mode-lst" onclick="trocarModo('lst')">
          🌡️ Confirmação LST (Calor)
        </button>
        <button class="mode-btn" id="mode-risco" onclick="trocarModo('risco')">
          ⚠️ Risco Integrado 2D
        </button>
      </div>

      <!-- Seletor de K (Ativo no modo K-Medoids) -->
      <div class="k-selector-container" id="container-k-selector">
        <span class="k-selector-label">K:</span>
        <button class="k-btn" id="btn-k2" onclick="trocarK(2)">2</button>
        <button class="k-btn" id="btn-k3" onclick="trocarK(3)">3</button>
        <button class="k-btn active" id="btn-k4" onclick="trocarK(4)">4 (Ótimo)</button>
        <button class="k-btn" id="btn-k5" onclick="trocarK(5)">5</button>
        <button class="k-btn" id="btn-k6" onclick="trocarK(6)">6</button>
      </div>

      <!-- Filtro Rápido de Cluster (Destacar 1 Cluster e Escurecer os demais) -->
      <div class="cluster-filter-container" id="container-cluster-filter">
        <span class="k-selector-label">Cluster:</span>
        <select id="select-cluster-filter" onchange="selecionarFiltroCluster(this.value)">
          <option value="todos">Todos os Clusters</option>
        </select>
        <button class="btn-clear-cluster" id="btn-clear-cluster" onclick="toggleFiltroCluster(null)" style="display: none;" title="Mostrar todos os bairros">✕ Ver Todos</button>
      </div>

      <div class="search-box">
        <input type="text" id="bairro-search" placeholder="🔍 Buscar bairro...">
      </div>

      <!-- Acesso Direto às Figuras do Relatório (Padrão Figura 3.4.6) -->
      <a href="figuras_relatorio/exportador_figuras_png.html" target="_blank" class="mode-btn" style="text-decoration: none; display: inline-flex; align-items: center; gap: 5px; background: #0284c7; color: #ffffff; font-weight: 700; border-radius: 6px; padding: 6px 12px; font-size: 0.78rem; transition: background 0.15s ease;" title="Abrir pranchas cartográficas limpas prontas para copiar/salvar em PNG para o relatório">
        📸 Figuras P/ Relatório (PNG)
      </a>
    </div>
  </header>

  <div id="main-container">
    <div id="map"></div>
    
    <div id="sidebar">
      <div id="sidebar-content">
        <div class="empty-state">
          <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"/>
          </svg>
          <p style="font-weight: 600; font-size: 0.95rem; margin-bottom: 6px;">Selecione um bairro no mapa</p>
          <p style="font-size: 0.8rem; color: #94a3b8;">Clique em qualquer polígono para inspecionar os 7 indicadores de vulnerabilidade, o arquétipo medoide e a temperatura de superfície observada (LST).</p>
        </div>
      </div>
    </div>
  </div>

  <script>
    // Injeção dos dados pelo backend Python
    const geojsonData = {json.dumps(geojson_final, ensure_ascii=False)};
    const configClustersPorK = {json.dumps(config_clusters_por_k, ensure_ascii=False)};
    let kAtual = {k_ativo};
    let modoAtual = 'kmedoids'; // 'kmedoids', 'lst' ou 'risco'
    let clusterFiltroAtivo = null; // null = todos, ou número do cluster (ex: 1, 2, 3...)
    let geojsonLayer = null;
    let selectedLayer = null;
    let legendControl = null;
    let medoidMarkersLayer = L.layerGroup();

    // Inicialização do Mapa Leaflet
    const map = L.map('map', {{
      center: [-22.915, -43.43],
      zoom: 11,
      zoomControl: true
    }});

    // Camadas de fundo limpas e sem marca d'água / sem API key (como no Ward)
    const esriLightGray = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
      maxZoom: 16
    }}).addTo(map);

    const osmStandard = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19
    }});

    L.control.layers({{
      "Fundo Neutro / Sem Ruído (Esri Gray)": esriLightGray,
      "OpenStreetMap (Ruas & Detalhes)": osmStandard
    }}, null, {{ position: 'topright' }}).addTo(map);

    medoidMarkersLayer.addTo(map);

    // Função de Estilo Dinâmica
    function getFeatureStyle(feature) {{
      const prop = feature.properties;
      let cor = '#94a3b8';
      let isMedoide = false;
      let fillOp = 0.80;
      let strokeColor = '#ffffff';
      let strokeW = 1.2;

      if (modoAtual === 'kmedoids') {{
        const kConfig = prop.clusters_por_k[kAtual.toString()];
        const cId = kConfig ? kConfig.cluster_id : null;
        isMedoide = kConfig ? kConfig.is_medoide : false;

        if (clusterFiltroAtivo !== null) {{
          if (cId === clusterFiltroAtivo) {{
            // Bairro do cluster selecionado: Cor viva em destaque
            cor = kConfig ? kConfig.cor : '#94a3b8';
            fillOp = 0.88;
            strokeColor = isMedoide ? '#000000' : '#ffffff';
            strokeW = isMedoide ? 2.5 : 1.4;
          }} else {{
            // Todos os outros bairros: CINZA suave / translúcido
            cor = '#94a3b8';
            fillOp = 0.22;
            strokeColor = '#cbd5e1';
            strokeW = 0.6;
          }}
        }} else {{
          // Sem filtro: visualização normal de todos os clusters
          cor = kConfig ? kConfig.cor : '#94a3b8';
          strokeColor = isMedoide ? '#000000' : '#ffffff';
          strokeW = isMedoide ? 2.5 : 1.2;
        }}
      }} else if (modoAtual === 'lst') {{
        cor = prop.cor_lst;
      }} else if (modoAtual === 'risco') {{
        cor = prop.cor_prioridade;
      }}

      return {{
        fillColor: cor,
        weight: strokeW,
        opacity: 1,
        color: strokeColor,
        fillOpacity: fillOp,
        dashArray: isMedoide ? '4, 4' : null
      }};
    }}

    // Interatividade com os polígonos
    function onEachFeature(feature, layer) {{
      layer.on({{
        mouseover: function(e) {{
          if (layer !== selectedLayer) {{
            const prop = layer.feature.properties;
            const kConfig = prop.clusters_por_k ? prop.clusters_por_k[kAtual.toString()] : null;
            const cId = kConfig ? kConfig.cluster_id : null;
            const isFiltroAtivo = (modoAtual === 'kmedoids' && clusterFiltroAtivo !== null);
            const pertenceCluster = (cId === clusterFiltroAtivo);

            layer.setStyle({{
              weight: 2.5,
              color: '#38bdf8',
              fillOpacity: (isFiltroAtivo && !pertenceCluster) ? 0.45 : 0.95
            }});
            layer.bringToFront();
          }}
        }},
        mouseout: function(e) {{
          if (layer !== selectedLayer) {{
            geojsonLayer.resetStyle(layer);
          }}
        }},
        click: function(e) {{
          selecionarBairro(feature, layer);
        }}
      }});
    }}

    // Atualização dos Marcadores de Medoides (Pins Arquetípicos)
    function atualizarMarcadoresMedoides() {{
      medoidMarkersLayer.clearLayers();
      if (modoAtual !== 'kmedoids') return;

      const kConfig = configClustersPorK[kAtual.toString()];

      geojsonLayer.eachLayer(function(layer) {{
        const p = layer.feature.properties;
        const kInfo = p.clusters_por_k[kAtual.toString()];
        if (kInfo && kInfo.is_medoide) {{
          // Se houver um filtro de cluster ativo, só exibe o medoide do cluster selecionado
          if (clusterFiltroAtivo !== null && kInfo.cluster_id !== clusterFiltroAtivo) {{
            return;
          }}

          const center = layer.getBounds().getCenter();
          const marker = L.circleMarker(center, {{
            radius: 7,
            fillColor: '#fbbf24',
            color: '#000000',
            weight: 2,
            opacity: 1,
            fillOpacity: 1
          }}).bindTooltip(`★ Medoide do Cluster ${{kInfo.cluster_id}}: ${{p.bairro}}`, {{
            permanent: false,
            direction: 'top'
          }});
          medoidMarkersLayer.addLayer(marker);
        }}
      }});
    }}

    // Funções de Filtro e Isolamento de Cluster
    function atualizarOpcoesDropdownCluster() {{
      const select = document.getElementById('select-cluster-filter');
      if (!select) return;

      const kConfig = configClustersPorK[kAtual.toString()] || {{}};
      let html = '<option value="todos">Todos os Clusters</option>';

      for (const [cId, info] of Object.entries(kConfig)) {{
        const selected = (clusterFiltroAtivo === parseInt(cId)) ? 'selected' : '';
        html += `<option value="${{cId}}" ${{selected}}>Cluster ${{cId}} (${{info.qtd}} bairros)</option>`;
      }}

      select.innerHTML = html;
      const btnClear = document.getElementById('btn-clear-cluster');
      if (btnClear) {{
        btnClear.style.display = clusterFiltroAtivo !== null ? 'inline-block' : 'none';
      }}
    }}

    function selecionarFiltroCluster(val) {{
      if (val === 'todos' || val === null || val === undefined) {{
        clusterFiltroAtivo = null;
      }} else {{
        clusterFiltroAtivo = parseInt(val);
      }}

      aplicarFiltroCluster();
    }}

    function toggleFiltroCluster(cId) {{
      if (modoAtual !== 'kmedoids') return;

      if (cId === null || clusterFiltroAtivo === cId) {{
        clusterFiltroAtivo = null;
      }} else {{
        clusterFiltroAtivo = cId;
      }}

      const select = document.getElementById('select-cluster-filter');
      if (select) {{
        select.value = clusterFiltroAtivo === null ? 'todos' : clusterFiltroAtivo.toString();
      }}

      aplicarFiltroCluster();
    }}

    function aplicarFiltroCluster() {{
      const btnClear = document.getElementById('btn-clear-cluster');
      if (btnClear) {{
        btnClear.style.display = clusterFiltroAtivo !== null ? 'inline-block' : 'none';
      }}

      geojsonLayer.eachLayer(function(layer) {{
        layer.setStyle(getFeatureStyle(layer.feature));
      }});

      atualizarMarcadoresMedoides();
      atualizarLegenda();
    }}

    // Construção da Legenda Dinâmica com Filtro Interativo por Clique
    function atualizarLegenda() {{
      if (legendControl) {{
        map.removeControl(legendControl);
      }}

      legendControl = L.control({{ position: 'bottomleft' }});
      legendControl.onAdd = function() {{
        const div = L.DomUtil.create('div', 'legend-control');

        if (modoAtual === 'kmedoids') {{
          const kConfig = configClustersPorK[kAtual.toString()];
          let btnClearLegenda = clusterFiltroAtivo !== null
            ? `<button class="btn-legend-clear" onclick="toggleFiltroCluster(null)" title="Mostrar todos os clusters">✕ Ver Todos</button>`
            : '';

          let html = `
            <div class="legend-header">
              <span>Clusters de Vulnerabilidade (K = ${{kAtual}})</span>
              ${{btnClearLegenda}}
            </div>
            <div style="font-size: 0.70rem; color: #64748b; margin-bottom: 6px; font-style: italic;">
              💡 Clique em um cluster para isolar seus bairros
            </div>
          `;

          for (const [cIdStr, info] of Object.entries(kConfig)) {{
            const cId = parseInt(cIdStr);
            const isAtivo = (clusterFiltroAtivo === cId);
            const isApagado = (clusterFiltroAtivo !== null && !isAtivo);
            let classesItem = 'legend-item';
            if (isAtivo) classesItem += ' item-ativo';
            if (isApagado) classesItem += ' item-apagado';

            html += `
              <div class="${{classesItem}}" onclick="toggleFiltroCluster(${{cId}})" title="Filtrar apenas Cluster ${{cId}}">
                <div class="legend-color" style="background: ${{info.cor}};"></div>
                <div class="legend-label">
                  Cluster ${{cId}}: ${{info.nome.split(' (')[0]}} 
                  <span class="legend-count">(${{info.qtd}} bairros)</span>
                </div>
              </div>
            `;
          }}
          div.innerHTML = html;
        }} else if (modoAtual === 'lst') {{
          div.innerHTML = `
            <div class="legend-header">
              <span>Confirmação Térmica: LST (°C)</span>
            </div>
            <div class="legend-item"><div class="legend-color" style="background: #2ca02c;"></div><div class="legend-label">&lt; 34.0°C (Refúgio Térmico)</div></div>
            <div class="legend-item"><div class="legend-color" style="background: #85e085;"></div><div class="legend-label">34.0 - 37.0°C (Ameno)</div></div>
            <div class="legend-item"><div class="legend-color" style="background: #fed976;"></div><div class="legend-label">37.0 - 39.5°C (Moderado)</div></div>
            <div class="legend-item"><div class="legend-color" style="background: #fd8d3c;"></div><div class="legend-label">39.5 - 41.5°C (Calor Elevado)</div></div>
            <div class="legend-item"><div class="legend-color" style="background: #f03b20;"></div><div class="legend-label">41.5 - 43.5°C (Ilha de Calor)</div></div>
            <div class="legend-item"><div class="legend-color" style="background: #bd0026;"></div><div class="legend-label">&gt; 43.5°C (Calor Extremo)</div></div>
          `;
        }} else if (modoAtual === 'risco') {{
          div.innerHTML = `
            <div class="legend-header">
              <span>Matriz 2D de Risco Integrado</span>
            </div>
            <div class="legend-item"><div class="legend-color" style="background: #dc2626;"></div><div class="legend-label">Urgente / Intervenção Imediata</div></div>
            <div class="legend-item"><div class="legend-color" style="background: #ea580c;"></div><div class="legend-label">Alerta Físico / Infraestrutura</div></div>
            <div class="legend-item"><div class="legend-color" style="background: #eab308;"></div><div class="legend-label">Acompanhamento Prioritário</div></div>
            <div class="legend-item"><div class="legend-color" style="background: #2563eb;"></div><div class="legend-label">Acompanhamento / Monitoramento</div></div>
            <div class="legend-item"><div class="legend-color" style="background: #16a34a;"></div><div class="legend-label">Baixo Risco / Preservação</div></div>
          `;
        }}

        return div;
      }};
      legendControl.addTo(map);
    }}

    // Troca de Modo de Visualização
    function trocarModo(novoModo) {{
      modoAtual = novoModo;
      document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
      document.getElementById(`mode-${{novoModo}}`).classList.add('active');

      const kSelector = document.getElementById('container-k-selector');
      const clusterSelector = document.getElementById('container-cluster-filter');

      if (novoModo === 'kmedoids') {{
        kSelector.style.opacity = '1';
        kSelector.style.pointerEvents = 'auto';
        if (clusterSelector) {{
          clusterSelector.style.opacity = '1';
          clusterSelector.style.pointerEvents = 'auto';
        }}
      }} else {{
        kSelector.style.opacity = '0.4';
        kSelector.style.pointerEvents = 'none';
        if (clusterSelector) {{
          clusterSelector.style.opacity = '0.4';
          clusterSelector.style.pointerEvents = 'none';
        }}
      }}

      geojsonLayer.eachLayer(function(layer) {{
        layer.setStyle(getFeatureStyle(layer.feature));
      }});

      atualizarMarcadoresMedoides();
      atualizarLegenda();
    }}

    // Troca Modular de K no HTML
    function trocarK(novoK) {{
      kAtual = novoK;
      clusterFiltroAtivo = null; // Reseta o filtro ao trocar o número de clusters

      document.querySelectorAll('.k-btn').forEach(btn => btn.classList.remove('active'));
      document.getElementById(`btn-k${{novoK}}`).classList.add('active');

      atualizarOpcoesDropdownCluster();

      geojsonLayer.eachLayer(function(layer) {{
        layer.setStyle(getFeatureStyle(layer.feature));
      }});

      atualizarMarcadoresMedoides();
      atualizarLegenda();

      if (selectedLayer) {{
        selecionarBairro(selectedLayer.feature, selectedLayer);
      }}
    }}

    // Inicialização da Camada GeoJSON
    geojsonLayer = L.geoJSON(geojsonData, {{
      style: getFeatureStyle,
      onEachFeature: onEachFeature
    }}).addTo(map);

    atualizarOpcoesDropdownCluster();
    atualizarMarcadoresMedoides();
    atualizarLegenda();

    // Renderização do Painel Lateral
    function selecionarBairro(feature, layer) {{
      if (selectedLayer) {{
        geojsonLayer.resetStyle(selectedLayer);
      }}
      selectedLayer = layer;
      layer.setStyle({{
        weight: 3.5,
        color: '#0f172a',
        fillOpacity: 0.95
      }});
      layer.bringToFront();

      const p = feature.properties;
      const kInfo = p.clusters_por_k[kAtual.toString()];

      let prioridadeCor = '#64748b';
      let prioridadeBg = '#f1f5f9';
      if (p.nivel_prioridade.includes('Urgente')) {{
        prioridadeCor = '#dc2626'; prioridadeBg = '#fee2e2';
      }} else if (p.nivel_prioridade.includes('Alerta')) {{
        prioridadeCor = '#d97706'; prioridadeBg = '#fef3c7';
      }} else if (p.nivel_prioridade.includes('Acompanhamento Prioritário')) {{
        prioridadeCor = '#b45309'; prioridadeBg = '#fef3c7';
      }} else if (p.nivel_prioridade.includes('Acompanhamento')) {{
        prioridadeCor = '#2563eb'; prioridadeBg = '#dbeafe';
      }} else {{
        prioridadeCor = '#16a34a'; prioridadeBg = '#dcfce7';
      }}

      const html = `
        <div class="sidebar-header">
          <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
              <div class="bairro-nome">${{p.bairro}}</div>
              <div class="ra-nome">${{p.regiao_administrativa}} • Ranking #${{p.ranking_prioridade}}</div>
            </div>
            <div style="background: ${{kInfo.cor}}; width: 20px; height: 20px; border-radius: 4px; border: 1px solid #000;" title="Cor do Cluster Ativo"></div>
          </div>
          <div style="margin-top: 6px; display: flex; flex-wrap: wrap; gap: 6px;">
            <span class="badge" style="background: ${{prioridadeBg}}; color: ${{prioridadeCor}};">${{p.nivel_prioridade}}</span>
            ${{kInfo.is_medoide ? '<span class="badge-medoide">★ MEDOIDE ARQUÉTIPO</span>' : ''}}
          </div>
        </div>

        <div class="sidebar-body">
          <div class="info-section">
            <h3>Cluster de Vulnerabilidade (K = ${{kAtual}})</h3>
            <p style="font-weight: 700; color: #0f172a; margin-bottom: 2px;">${{kInfo.cluster_nome}}</p>
            <p style="font-size: 0.8rem; color: #475569;">Arquétipo Concreto: <b>${{kInfo.medoide}}</b></p>
          </div>

          <div class="metric-grid">
            <div class="metric-card destaque-calor">
              <div class="metric-label">LST (Calor Real)</div>
              <div class="metric-value">${{p.lst_media.toFixed(1)}} °C</div>
            </div>
            <div class="metric-card destaque-vuln">
              <div class="metric-label">Vulnerabilidade (0-100)</div>
              <div class="metric-value">${{p.score_vulnerabilidade_100.toFixed(1)}}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Risco Integrado 2D</div>
              <div class="metric-value">${{p.score_risco_integrado.toFixed(1)}}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">População Total</div>
              <div class="metric-value">${{p.pop.toLocaleString('pt-BR')}}</div>
            </div>
          </div>

          <div class="features-list">
            <div class="features-title">7 Dimensões de Vulnerabilidade Socioambiental</div>
            
            <div class="feat-row">
              <span class="feat-name">Renda Mediana:</span>
              <div style="display: flex; align-items: center;">
                <span class="feat-val">R$ ${{p.renda_resp_mediana.toLocaleString('pt-BR', {{maximumFractionDigits: 0}})}}</span>
              </div>
            </div>

            <div class="feat-row">
              <span class="feat-name">% População em Favela:</span>
              <div style="display: flex; align-items: center;">
                <span class="feat-val">${{p.pct_pop_favela.toFixed(1)}}%</span>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ${{p.pct_pop_favela}}%; background: #ea580c;"></div></div>
              </div>
            </div>

            <div class="feat-row">
              <span class="feat-name">Densidade Habitada:</span>
              <span class="feat-val">${{Math.round(p.densidade_ajustada).toLocaleString('pt-BR')}} hab/km²</span>
            </div>

            <div class="feat-row">
              <span class="feat-name">Arborização Integrada:</span>
              <div style="display: flex; align-items: center;">
                <span class="feat-val">${{p.arborizacao_integrada.toFixed(1)}}%</span>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ${{p.arborizacao_integrada}}%; background: #16a34a;"></div></div>
              </div>
            </div>

            <div class="feat-row">
              <span class="feat-name">Sem Saneamento (Água/Esgoto):</span>
              <div style="display: flex; align-items: center;">
                <span class="feat-val">${{p.pct_sem_saneamento.toFixed(1)}}%</span>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ${{Math.min(100, p.pct_sem_saneamento * 2)}}%; background: #dc2626;"></div></div>
              </div>
            </div>

            <div class="feat-row">
              <span class="feat-name">Pavimentação do Solo:</span>
              <div style="display: flex; align-items: center;">
                <span class="feat-val">${{p.pct_pavimentacao.toFixed(1)}}%</span>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ${{p.pct_pavimentacao}}%; background: #64748b;"></div></div>
              </div>
            </div>

            <div class="feat-row">
              <span class="feat-name">Taxa de Analfabetismo:</span>
              <div style="display: flex; align-items: center;">
                <span class="feat-val">${{p.taxa_analfabetismo.toFixed(1)}}%</span>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ${{Math.min(100, p.taxa_analfabetismo * 8)}}%; background: #b45309;"></div></div>
              </div>
            </div>
          </div>

          <div class="info-section" style="border-left-color: #10b981;">
            <h3>Diretriz Estratégica Recomendada</h3>
            <p>${{p.acao_recomendada}}</p>
          </div>

          ${{p.alerta_maup && p.alerta_maup.includes('Nível 2') ? `
            <div class="maup-alert">
              <b>⚠️ ALERTA MAUP (NÍVEL 2):</b> Recomenda-se microanálise setorial intra-bairro devido à heterogeneidade socioespacial severa na área urbanizada.
            </div>
          ` : ''}}
        </div>
      `;

      document.getElementById('sidebar-content').innerHTML = html;
    }}

    // Busca Interativa de Bairros
    document.getElementById('bairro-search').addEventListener('input', function(e) {{
      const query = e.target.value.toLowerCase().trim();
      if (!query) return;

      geojsonLayer.eachLayer(function(layer) {{
        const nome = layer.feature.properties.bairro.toLowerCase();
        if (nome === query || (query.length >= 3 && nome.startsWith(query))) {{
          map.fitBounds(layer.getBounds(), {{ maxZoom: 14, animate: true }});
          selecionarBairro(layer.feature, layer);
        }}
      }});
    }});
  </script>
</body>
</html>
"""

    with open(caminho_saida_html, "w", encoding="utf-8") as f:
        f.write(conteudo_html)

    print(f"\n[OK] Mapa Interativo HTML gerado com sucesso em:\n     -> {caminho_saida_html}")
    print("=" * 70 + "\n")

    return caminho_saida_html


if __name__ == "__main__":
    k_arg = 4
    if len(sys.argv) > 1:
        try:
            k_arg = int(sys.argv[1])
        except ValueError:
            pass
    gerar_mapa_html_kmedoids(k_ativo=k_arg)
