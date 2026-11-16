# Seleção de Variáveis e Limiares – Decisões do Projeto

## Contexto

A seleção de variáveis parte dos 162 bairros do Rio já agrupados pelo K-Medoids (K=4; clusters de 26, 45, 73 e 18 bairros) e gera dois subconjuntos de 5 variáveis cada, usados no clustering da Sprint 3.

As candidatas são os 7 indicadores de vulnerabilidade do K-Medoids: renda, % em favela, densidade, arborização, % sem saneamento, % pavimentação e analfabetismo. Escores, ranking e distâncias ficaram de fora porque derivam dos clusters e criariam circularidade. A temperatura de superfície (`lst_media`) é variável de controle: não é testada nem removida, só acompanha os arquivos para validação.

São duas seleções com objetivos distintos: **A, relevância (η²)** e **B, redundância (Spearman)**.

## Critérios e limiares

Cada seleção usa um único limiar, partindo de uma convenção e depois testado contra outros valores (seção de sensibilidade).

| Seleção | Pergunta | Estatística | Limiar | Regra |
| --- | --- | --- | --- | --- |
| A – Relevância | A variável separa os clusters? | η² = SS_entre / SS_total, por `cluster_id` | η² ≥ 0,14 | Mantém quem passa |
| B – Redundância | A variável repete outra? | Correlação de Spearman entre pares | \|ρ\| ≥ 0,70 | Poda gulosa: remove uma variável por par redundante |

**Por que η² ≥ 0,14.** É o corte de efeito grande de Cohen. Abaixo disso, a variável explica menos de 14% da variabilidade entre bairros por cluster e pouco contribui para distinguir os grupos.

**Por que |ρ| ≥ 0,70.** É o patamar usual de correlação forte. Spearman, e não Pearson, porque mede a relação monótona por postos e não depende de linearidade nem de outliers.

**Como a poda desempata.** Enquanto houver par com |ρ| ≥ 0,70, pega o par de maior |ρ| e remove a variável com maior |ρ| médio contra as demais, isto é, a mais "explicada" pelo resto da base. Repete até não sobrar par acima do limiar.

**Tratamento de borda.** Valores ausentes ficam fora de todas as somas do η²; correlação NaN (coluna constante) conta como zero. Nas 7 candidatas não há ausentes.

## Resultado por variável

Cada seleção ficou com 5 das 7 candidatas, mas não as mesmas: a A (η²) mantém renda e analfabetismo, a B (Spearman) mantém saneamento e pavimentação.

| Variável | η² | A – η² (≥ 0,14) | B – Spearman (\|ρ\| < 0,70) |
| --- | --- | --- | --- |
| Arborização integrada | 0,759 | Mantida | Mantida |
| Renda mediana do responsável | 0,758 | Mantida | Removida (ρ = 0,74 com arborização) |
| % população em favela | 0,730 | Mantida | Mantida |
| Taxa de analfabetismo | 0,518 | Mantida | Removida (ρ = −0,81 com renda) |
| Densidade ajustada | 0,181 | Mantida | Mantida |
| % pavimentação | 0,054 | Removida | Mantida |
| % sem saneamento | 0,006 | Removida | Mantida |
| LST média (controle) | 0,150 | Fora da seleção | Fora da seleção |

**Passos da poda de Spearman.** Três pares passam de 0,70: renda–analfabetismo (−0,81), renda–arborização (0,74) e arborização–analfabetismo (−0,73).

1. Maior par: renda–analfabetismo. O analfabetismo tem |ρ| médio maior (3,30 contra 2,94, somando contra as outras 6) e sai.
2. Sem o analfabetismo, resta renda–arborização (0,74). Renda soma 2,13 e arborização 2,11, então renda sai.
3. Nenhum par fica acima de 0,70 e a poda para.

**Pontos de atenção**

- O passo 2 é um empate técnico (diferença de 0,02). Com outro desempate, arborização sairia e renda ficaria, e o conjunto B mudaria.
- Pavimentação e saneamento têm η² quase nulo: não discriminam os clusters, então a seleção B mantém duas variáveis pouco úteis para separar grupos. É o contraste que justifica comparar A e B na Sprint 3.
- A LST (η² = 0,150) ultrapassaria o limiar, mas fica fora de propósito: é a referência térmica com que se valida o agrupamento, e usar a variável para agrupar e depois validá-la seria circular.
- O η² de densidade (0,181) fica perto do corte de 0,14; a seção de sensibilidade mostra até onde o corte pode subir.

## Parâmetros da etapa seguinte

O `clustering_sprint3.py` consome os dois conjuntos e fixa os demais parâmetros abaixo.

| Parâmetro | Valor | Observação |
| --- | --- | --- |
| Variáveis por seleção | 5 | Os scripts avisam se uma seleção não fechar em 5 |
| K (K-Medoids e AGNES) | 3, 4, 5 | Distâncias Manhattan e euclidiana; linkage average e complete |
| DBSCAN | eps 0,25 e 0,28; minPts 3 e 5 | Aplicado em dados Min-Max |
| Validação | Sem folds | Cada cenário é ajustado nos 162 bairros; semente 42 |
| Estabilidade (ARI) | 100 reamostragens de 80% dos bairros | Sem reposição; ruído (-1) fica fora do ARI |
| Pré-processamento | Min-Max e imputação pela mediana | Ajustados nos 162 bairros |

Total de 32 cenários: 12 de K-Medoids, 12 de AGNES e 8 de DBSCAN. Os valores de eps, minPts e K foram fixados no plano de experimentação; o código não registra como foram escolhidos.

## Sensibilidade dos limiares

Os dois limiares ficam dentro de faixas em que a seleção não muda: η² entre 0,06 e 0,18 e |ρ| entre 0,60 e 0,74. Testei outros cortes sobre os mesmos 162 bairros.

**η² (seleção A)**

| Corte de η² | Variáveis mantidas | O que muda |
| --- | --- | --- |
| 0,01 | 6 | Entra pavimentação (η² = 0,054) |
| 0,06 a 0,18 | 5 | Faixa estável; inclui 0,10 e 0,14 |
| 0,20 a 0,50 | 4 | Sai densidade (η² = 0,181) |
| 0,60 | 3 | Sai também analfabetismo (η² = 0,518) |

**Spearman (seleção B)**

| Corte de \|ρ\| | Variáveis mantidas | Removidas |
| --- | --- | --- |
| 0,50 a 0,55 | 4 | Analfabetismo, renda e % favela |
| 0,60 a 0,70 | 5 | Analfabetismo e renda |
| 0,75 a 0,80 | 6 | Só analfabetismo |
| 0,85 | 7 | Nenhuma |

**Leitura.** O 0,14 do η² fica no meio de uma faixa de 0,06 a 0,18, então a convenção não decide sozinha: qualquer corte nessa faixa dá as mesmas 5 variáveis. O 0,70 do Spearman fica no topo de uma faixa de 0,60 a 0,74, e o par renda–arborização (0,74) é quem o separa da faixa seguinte, de 6 variáveis. O 0,70 não é o mais seguro: 0,65 daria o mesmo conjunto com mais folga.

**Limite do teste.** Ele mostra que a seleção é estável, não que ela é a melhor para o clustering. Isso só se responde comparando as métricas de silhueta, Davies-Bouldin e ARI dos 32 cenários da Sprint 3.
