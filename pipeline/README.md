# 🏙️ Pipeline K-Medoids V2: Vulnerabilidade Socioespacial & Ilhas de Calor (Rio de Janeiro)

Esta pasta contém o **ciclo completo de Ciência de Dados (6 etapas)** da modelagem de agrupamento não supervisionado por **K-Medoids (algoritmo PAM com Distância de Manhattan $L_1$)** para os 162 bairros do município do Rio de Janeiro.

A pasta é **100% autônoma e independente**: qualquer pessoa pode baixá-la do Google Drive para o seu computador e executar os 6 scripts sequencialmente, do primeiro ao último, sem necessidade de caminhos externos ou configurações complexas.

---

## 📁 Estrutura da Pasta

```text
etapas_kmedoids_v2/
├── dados/                                     # Tabelas de entrada brutas necessárias
│   ├── base_bairro_estatica.csv               # Base cadastral de 162 bairros com 10 atributos
│   └── ivs_socioeconomico_bairros_rio.geojson # Geometria vetorial oficial dos bairros
├── 01_ingestao_limpeza.py                     # Etapa 1: Ingestão, auditoria e imputação
├── 02_eda_selecao.py                          # Etapa 2: EDA, aglutinações e engenharia de features
├── 03_preprocessamento_kmedoids.py            # Etapa 3: Normalização balanceada Min-Max [0, 1]
├── 04_modelagem_kmedoids.py                   # Etapa 4: Algoritmo PAM (L1), varredura K=2..6 e K=4 ótimo
├── 05_perfilamento_matriz_kmedoids.py         # Etapa 5: Perfilamento, Decomposição Eta² e Matriz de Risco 2D
├── 06_geracao_outputs_kmedoids.py             # Etapa 6: Consolidação tabular final e acionamento dos mapas
├── gerar_mapa_html_kmedoids.py                # Auxiliar: Geração do Mapa Interativo Leaflet
├── gerar_figuras_mapa_relatorio.py            # Auxiliar: Geração das pranchas e exportador PNG
└── README.md                                  # Este guia explicativo
```

> **Nota:** A pasta `outputs/` não vem pré-populada; ela será criada automaticamente assim que o primeiro script for executado.

---

## ⚙️ Pré-requisitos (Bibliotecas Python)

O pipeline foi projetado para exigir apenas as bibliotecas essenciais de análise de dados:

```bash
pip install pandas numpy
```

*(Não é necessário instalar bibliotecas GIS pesadas como GDAL ou GeoPandas para rodar os 6 scripts).*

---

## 🚀 Como Executar (Passo a Passo: 1 por 1)

Abra o terminal (ou Prompt de Comando / PowerShell), navegue até esta pasta e execute os arquivos na ordem sequencial:

```bash
# 1. Ingestão, auditoria de nulos e imputação censitária (Joá / RA 24)
python3 01_ingestao_limpeza.py

# 2. Engenharia de atributos (arborizacao_integrada e pct_sem_saneamento) e reserva da LST
python3 02_eda_selecao.py

# 3. Normalização Min-Max balanceada nas 7 variáveis de vulnerabilidade
python3 03_preprocessamento_kmedoids.py

# 4. Modelagem PAM com métrica de Manhattan, cálculo de TD, Silhueta, DB e Bootstrap ARI
python3 04_modelagem_kmedoids.py

# 5. Perfilamento dos medoides reais, Decomposição do Eta² e Matriz de Risco 2D
python3 05_perfilamento_matriz_kmedoids.py

# 6. Geração dos relatórios finais tabulares, mapa interativo e figuras do relatório
python3 06_geracao_outputs_kmedoids.py
```

---

## 📊 O Que Cada Etapa Produz na Pasta `outputs/`

Ao concluir os 6 passos, a pasta `outputs/` conterá:

1. **Tabelas Analíticas Finais**:
   - `classificacao_bairros_ilhas_calor_kmedoids.csv`: Classificação completa dos 162 bairros com ranking de risco integrado, cluster_id, medoide arquetípico, escores e alerta de MAUP.
   - `perfil_resumo_clusters_kmedoids.csv`: Resumo executivo dos 4 clusters com médias sociodemográficas, térmicas e diretrizes de políticas públicas.
   - `metricas_k_otimo_kmedoids.csv`: Tabela comparativa de métricas ($TD$, Silhueta, $DB$, $ARI$) para $K \in [2, 6]$.
   - `eta_quadrado_atributos_kmedoids.csv`: Decomposição da variância explicada ($\eta^2$).

2. **Visualizações Cartográficas**:
   - `mapa_ilhas_calor_kmedoids.html`: Mapa interativo Leaflet com seletor de $K$, isolamento de clusters e alternância para o mapa de calor térmico real (LST).
   - `figuras_relatorio/`: Pasta com as pranchas vetoriais SVG de cada cluster isolado e o arquivo `exportador_figuras_png.html`, permitindo baixar os mapas em PNG de 300 DPI para o relatório final.
