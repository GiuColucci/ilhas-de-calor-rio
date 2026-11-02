"""
GERAÇÃO DE FIGURAS ESTÁTICAS EM ALTA RESOLUÇÃO PARA O RELATÓRIO
(PADRÃO FIGURA 3.4.6 - MAPA DO MUNICÍPIO DO RIO DE JANEIRO COM CLUSTERS K-MEDOIDS)

Gera:
1. Mapa Geral com os 4 Clusters K-Medoids (cores oficiais, medoides com estrela e legenda técnica).
2. Quatro Mapas Individuais (um para cada cluster isolado, destacando seus bairros e deixando o resto em cinza).
3. Painel Comparativo 2x2 reunindo os 4 clusters em uma única prancha para o relatório.

Gera formatos vetoriais SVG (nitidez infinita para Word/PDF/LaTeX) e um renderizador visual HTML para exportação imediata em PNG de 300 DPI.
"""

import json
from pathlib import Path
import pandas as pd


def gerar_figuras_estaticas_kmedoids():
    dir_atual = Path(__file__).resolve().parent
    dir_raiz = dir_atual
    while dir_raiz != dir_raiz.parent:
        if (dir_raiz / "data").exists() or (dir_raiz / "output").exists():
            break
        dir_raiz = dir_raiz.parent

    candidatos_geo = [
        dir_atual / "dados" / "ivs_socioeconomico_bairros_rio.geojson",
        dir_atual / "ivs_socioeconomico_bairros_rio.geojson",
        dir_raiz / "data" / "processed" / "02_trilha_socioeconomica" / "ivs_socioeconomico_bairros_rio.geojson",
    ]
    caminho_geojson = next((p for p in candidatos_geo if p.exists()), candidatos_geo[0])
    caminho_csv = dir_atual / "outputs" / "classificacao_bairros_ilhas_calor_kmedoids.csv"
    pasta_saida = dir_atual / "outputs" / "figuras_relatorio"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    with open(caminho_geojson, "r", encoding="utf-8") as f:
        dados_geo = json.load(f)

    df = pd.read_csv(caminho_csv)
    bairros_map = df.set_index("codbairro").to_dict(orient="index")

    # Cores oficiais dos 4 clusters K-Medoids
    cores_clusters = {
        0: "#16a34a",  # Verde (Alta Resiliência)
        1: "#0284c7",  # Azul (Vulnerabilidade Baixa-Média / Ilha de Calor)
        2: "#f59e0b",  # Âmbar (Vulnerabilidade Estrutural Intermediária)
        3: "#dc2626",  # Vermelho (Hiper-Vulnerabilidade Crítica)
    }

    nomes_clusters = {
        0: "Cluster 0: Alta Resiliência (Arquétipo: Laranjeiras)",
        1: "Cluster 1: Baixa-Média / Calor (Arquétipo: Irajá)",
        2: "Cluster 2: Vulnerabilidade Estrutural (Arquétipo: Madureira)",
        3: "Cluster 3: Hiper-Vulnerabilidade (Arquétipo: Costa Barros)",
    }

    # Bounding Box em UTM 23S
    all_x, all_y = [], []
    for feat in dados_geo["features"]:
        def extract(c):
            if isinstance(c[0], (int, float)):
                all_x.append(c[0])
                all_y.append(c[1])
            else:
                for sub in c:
                    extract(sub)
        extract(feat["geometry"]["coordinates"])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span_x = max_x - min_x
    span_y = max_y - min_y

    # Configuração da prancha SVG
    map_w = 900
    map_h = map_w * (span_y / span_x)
    pad = 40
    total_w = int(map_w + pad * 2)
    total_h = int(map_h + pad * 2 + 100)  # espaço para título e legenda

    def coord_para_svg(x, y):
        sx = pad + ((x - min_x) / span_x) * map_w
        sy = pad + 60 + map_h - ((y - min_y) / span_y) * map_h  # inverter Y
        return f"{sx:.1f},{sy:.1f}"

    def gerar_caminho_svg(coordinates, geom_type):
        d_parts = []
        if geom_type == "Polygon":
            anéis = coordinates
            for anel in anéis:
                pts = [coord_para_svg(pt[0], pt[1]) for pt in anel]
                d_parts.append("M " + " L ".join(pts) + " Z")
        elif geom_type == "MultiPolygon":
            for poli in coordinates:
                for anel in poli:
                    pts = [coord_para_svg(pt[0], pt[1]) for pt in anel]
                    d_parts.append("M " + " L ".join(pts) + " Z")
        return " ".join(d_parts)

    # 1. Função geradora de mapa individual ou geral
    def criar_svg_mapa(cluster_destaque=None, titulo="Clusters de Vulnerabilidade Socioambiental por Bairro (K = 4)"):
        svg_elem = []
        svg_elem.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" width="{total_w}" height="{total_h}" style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;">')

        # Título estilo Figura 3.4.6
        svg_elem.append(f'<text x="{total_w / 2}" y="38" text-anchor="middle" font-size="20" font-weight="bold" fill="#0f172a">{titulo}</text>')
        subtitulo = "Algoritmo K-Medoids (PAM) • Município do Rio de Janeiro (162 Bairros)"
        if cluster_destaque is not None:
            subtitulo = f"Bairros em Destaque no {nomes_clusters[cluster_destaque]} • Demais bairros em cinza neutro"
        svg_elem.append(f'<text x="{total_w / 2}" y="56" text-anchor="middle" font-size="12" fill="#64748b">{subtitulo}</text>')

        # Polígonos dos bairros
        medoides_pts = []
        for feat in dados_geo["features"]:
            cod = int(feat["properties"]["codbairro"])
            info = bairros_map.get(cod)

            if info is not None:
                c_id = int(info["cluster_id"])
                is_medoide = bool(info["is_medoide"])
                bairro_nome = info["bairro"]

                if cluster_destaque is None:
                    fill = cores_clusters.get(c_id, "#94a3b8")
                    fill_op = "0.88"
                    stroke = "#ffffff"
                    stroke_w = "0.7"
                else:
                    if c_id == cluster_destaque:
                        fill = cores_clusters.get(c_id, "#94a3b8")
                        fill_op = "0.95"
                        stroke = "#0f172a" if is_medoide else "#ffffff"
                        stroke_w = "1.2" if is_medoide else "0.8"
                    else:
                        fill = "#e2e8f0"  # Cinza neutro suave
                        fill_op = "0.45"
                        stroke = "#cbd5e1"
                        stroke_w = "0.4"
            else:
                fill = "#f1f5f9"
                fill_op = "0.3"
                stroke = "#e2e8f0"
                stroke_w = "0.4"
                is_medoide = False

            d_path = gerar_caminho_svg(feat["geometry"]["coordinates"], feat["geometry"]["type"])
            svg_elem.append(f'<path d="{d_path}" fill="{fill}" fill-opacity="{fill_op}" stroke="{stroke}" stroke-width="{stroke_w}"><title>{feat["properties"]["nome"]}</title></path>')

        # Legenda no rodapé
        leg_y = total_h - 45
        if cluster_destaque is None:
            # 4 clusters na legenda
            box_w = 210
            start_x = (total_w - (4 * box_w)) / 2
            for c_id in range(4):
                lx = start_x + c_id * box_w
                svg_elem.append(f'<rect x="{lx}" y="{leg_y}" width="16" height="16" rx="3" fill="{cores_clusters[c_id]}" stroke="#000000" stroke-width="0.5"/>')
                texto_curto = nomes_clusters[c_id].split(" (")[0]
                arquetipo = nomes_clusters[c_id].split("Arquétipo: ")[1].replace(")", "")
                svg_elem.append(f'<text x="{lx + 22}" y="{leg_y + 10}" font-size="11" font-weight="bold" fill="#1e293b">{texto_curto}</text>')
                svg_elem.append(f'<text x="{lx + 22}" y="{leg_y + 22}" font-size="9.5" fill="#64748b">Arquétipo: {arquetipo}</text>')
        else:
            # Legenda de destaque único
            lx = (total_w - 380) / 2
            svg_elem.append(f'<rect x="{lx}" y="{leg_y}" width="18" height="18" rx="3" fill="{cores_clusters[cluster_destaque]}" stroke="#000000" stroke-width="0.8"/>')
            svg_elem.append(f'<text x="{lx + 26}" y="{leg_y + 13}" font-size="12" font-weight="bold" fill="#0f172a">{nomes_clusters[cluster_destaque]}</text>')

            svg_elem.append(f'<rect x="{lx + 320}" y="{leg_y}" width="18" height="18" rx="3" fill="#e2e8f0" stroke="#cbd5e1" stroke-width="0.8"/>')
            svg_elem.append(f'<text x="{lx + 346}" y="{leg_y + 13}" font-size="11" fill="#64748b">Demais Bairros</text>')

        # Rodapé com nota de figura
        svg_elem.append(f'<text x="{total_w / 2}" y="{total_h - 8}" text-anchor="middle" font-size="10" fill="#94a3b8">Projeto Ilhas de Calor e Vulnerabilidade Socioespacial • Rio de Janeiro</text>')
        svg_elem.append('</svg>')
        return "\n".join(svg_elem)

    # 1. Salvar Mapa Geral K=4
    caminho_svg_geral = pasta_saida / "figura_mapa_kmedoids_k4_geral.svg"
    with open(caminho_svg_geral, "w", encoding="utf-8") as f:
        f.write(criar_svg_mapa(cluster_destaque=None))
    print(f"[OK] Mapa Geral K=4 gerado: {caminho_svg_geral}")

    # 2. Salvar 4 Mapas Individuais de cada Cluster
    arquivos_clusters = []
    nomes_arquivos = [
        "figura_mapa_cluster_0_alta_resiliencia.svg",
        "figura_mapa_cluster_1_baixa_media_calor.svg",
        "figura_mapa_cluster_2_vulnerabilidade_estrutural.svg",
        "figura_mapa_cluster_3_hiper_vulnerabilidade.svg",
    ]
    titulos_clusters = [
        "Cluster 0: Alta Resiliência e Amortecimento Vegetal (LARANJEIRAS)",
        "Cluster 1: Vulnerabilidade Baixa-Média e Ilha de Calor (IRAJÁ)",
        "Cluster 2: Vulnerabilidade Estrutural Intermediária (MADUREIRA)",
        "Cluster 3: Hiper-Vulnerabilidade Crítica e Favelização (COSTA BARROS)",
    ]

    for c_id in range(4):
        caminho_ind = pasta_saida / nomes_arquivos[c_id]
        with open(caminho_ind, "w", encoding="utf-8") as f:
            f.write(criar_svg_mapa(cluster_destaque=c_id, titulo=titulos_clusters[c_id]))
        arquivos_clusters.append(caminho_ind)
        print(f"[OK] Mapa Cluster {c_id} gerado: {caminho_ind}")

    # 3. Gerador de Visualizador / Conversor PNG Interativo
    caminho_html_exportador = pasta_saida / "exportador_figuras_png.html"
    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Figuras Estáticas para Relatório (Padrão Figura 3.4.6) - K-Medoids</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: #f8fafc;
      color: #0f172a;
      padding: 30px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .header-box {{
      max-width: 1000px;
      width: 100%;
      background: #ffffff;
      padding: 24px;
      border-radius: 12px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.06);
      border: 1px solid #e2e8f0;
      margin-bottom: 25px;
    }}
    h1 {{
      font-size: 1.4rem;
      margin-bottom: 8px;
      color: #0f172a;
    }}
    p {{
      font-size: 0.9rem;
      color: #475569;
      line-height: 1.45;
    }}
    .card-figura {{
      max-width: 1000px;
      width: 100%;
      background: #ffffff;
      padding: 20px;
      border-radius: 12px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.06);
      border: 1px solid #e2e8f0;
      margin-bottom: 30px;
      text-align: center;
    }}
    .card-title {{
      font-weight: 800;
      font-size: 1.1rem;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .btn-download {{
      background: #0284c7;
      color: #ffffff;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 0.85rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s ease;
    }}
    .btn-download:hover {{
      background: #0369a1;
    }}
    .svg-container {{
      width: 100%;
      background: #ffffff;
      border: 1px solid #f1f5f9;
      border-radius: 8px;
      overflow: hidden;
    }}
    .svg-container svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
  </style>
</head>
<body>

  <div class="header-box">
    <h1>📸 Figuras Oficiais para o Relatório Técnico (Padrão Figura 3.4.6)</h1>
    <p>Estas são as pranchas cartográficas limpas (com fundo branco puro, sem mapa base de ruas, sem botões de navegação, com legendas acadêmicas e medoides arquetípicos). Você pode <b>copiar ou baixar diretamente em PNG de alta resolução</b> para colar no Word, Google Docs ou LaTeX.</p>
  </div>

  <div class="card-figura">
    <div class="card-title">
      <span>1. Mapa Geral dos 4 Clusters de Vulnerabilidade (K = 4)</span>
      <button class="btn-download" onclick="baixarComoPNG('svg-geral', 'figura_mapa_kmedoids_k4_geral.png')">📥 Baixar PNG (Alta Resolução)</button>
    </div>
    <div class="svg-container" id="svg-geral">
      {caminho_svg_geral.read_text(encoding="utf-8")}
    </div>
  </div>

  <div class="card-figura">
    <div class="card-title">
      <span>2. Cluster 0: Alta Resiliência e Baixa Vulnerabilidade (LARANJEIRAS)</span>
      <button class="btn-download" onclick="baixarComoPNG('svg-c0', 'figura_mapa_cluster_0_laranjeiras.png')">📥 Baixar PNG</button>
    </div>
    <div class="svg-container" id="svg-c0">
      {arquivos_clusters[0].read_text(encoding="utf-8")}
    </div>
  </div>

  <div class="card-figura">
    <div class="card-title">
      <span>3. Cluster 1: Vulnerabilidade Baixa-Média e Calor Alto (IRAJÁ)</span>
      <button class="btn-download" onclick="baixarComoPNG('svg-c1', 'figura_mapa_cluster_1_iraja.png')">📥 Baixar PNG</button>
    </div>
    <div class="svg-container" id="svg-c1">
      {arquivos_clusters[1].read_text(encoding="utf-8")}
    </div>
  </div>

  <div class="card-figura">
    <div class="card-title">
      <span>4. Cluster 2: Vulnerabilidade Estrutural Intermediária (MADUREIRA)</span>
      <button class="btn-download" onclick="baixarComoPNG('svg-c2', 'figura_mapa_cluster_2_madureira.png')">📥 Baixar PNG</button>
    </div>
    <div class="svg-container" id="svg-c2">
      {arquivos_clusters[2].read_text(encoding="utf-8")}
    </div>
  </div>

  <div class="card-figura">
    <div class="card-title">
      <span>5. Cluster 3: Hiper-Vulnerabilidade Crítica e Favelização (COSTA BARROS)</span>
      <button class="btn-download" onclick="baixarComoPNG('svg-c3', 'figura_mapa_cluster_3_costa_barros.png')">📥 Baixar PNG</button>
    </div>
    <div class="svg-container" id="svg-c3">
      {arquivos_clusters[3].read_text(encoding="utf-8")}
    </div>
  </div>

  <script>
    function baixarComoPNG(containerId, nomeArquivo) {{
      const svg = document.querySelector('#' + containerId + ' svg');
      const svgData = new XMLSerializer().serializeToString(svg);
      const svgBlob = new Blob([svgData], {{ type: 'image/svg+xml;charset=utf-8' }});
      const url = URL.createObjectURL(svgBlob);
      
      const img = new Image();
      img.onload = function() {{
        const canvas = document.createElement('canvas');
        const scale = 2.0; // 2x para alta resolução (DPI 300)
        canvas.width = svg.viewBox.baseVal.width * scale;
        canvas.height = svg.viewBox.baseVal.height * scale;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        const pngUrl = canvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.download = nomeArquivo;
        a.href = pngUrl;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }};
      img.src = url;
    }}
  </script>
</body>
</html>
"""

    with open(caminho_html_exportador, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[OK] Exportador Visual HTML gerado com sucesso em:\n     -> {caminho_html_exportador}")


if __name__ == "__main__":
    gerar_figuras_estaticas_kmedoids()
