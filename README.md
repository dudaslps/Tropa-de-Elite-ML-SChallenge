# Tropa-de-Elite
# Modelo de Risco Ambiental e de Sinistro Mecânico — Máquinas Agrícolas

**SOMPO Challenge — Grupo TROPA DE ELITE**

Integrantes: Arthur Costa Donaire (RM571283) · Felipe Pereira de Jesus (RM573263) ·
Giovanna Pereira de Oliveira (RM570989) · Gustavo Paiva Silva (RM572249) ·
Maria Eduarda Soares Lopes e Souza (RM572612)

## Visão geral

Este notebook produz **duas previsões independentes, com nomes e propósitos diferentes** — a separação é deliberada, não uma pendência:

| | Risco ambiental | Risco de sinistro |
|---|---|---|
| Onde | Parte 1 (seções 1–13) | Parte 3 (seções 15–17) |
| Treinado com | `dataset_risco_agricola_final` (SISSER + IBGE + Open-Meteo) | `dataset_falha_mecanica_sintetico` (gerado na Parte 2) |
| O que estima | Quão desfavoráveis estão as condições climáticas da região/cultura naquele momento | Chance de falha mecânica/sinistro da máquina, a partir das leituras do sensor |
| É a saída recomendada para o firmware? | Não (informativa/complementar) | **Sim — é `probabilidade_sinistro`/`classificacao_ml`, sem combinação com o modelo de risco ambiental** |

O rótulo `sinistro` do SISSER marca **perda de produção agrícola por evento climático** (seca, granizo, geada), não pane/colisão/capotamento de trator — por isso a saída da Parte 1 é chamada de "risco ambiental" e não de "sinistro", para não confundir com o que a Parte 3 mede de fato.

**Fluxo do notebook, em ordem:**
1. Treina um classificador binário de risco ambiental a partir do dataset real (`dataset_risco_agricola_final`).
2. Lê o `.json`/`.jsonl` gerado pela simulação do ESP32 no Wokwi.
3. Aplica o modelo de risco ambiental a cada leitura e devolve `probabilidade_risco_ambiental`.
4. Gera o dataset sintético de falha mecânica e treina, separadamente, o modelo de risco de sinistro.
5. Aplica o modelo de risco de sinistro às mesmas leituras do ESP32 e devolve `probabilidade_sinistro` — esta é a saída final recomendada.

## Como executar

O notebook detecta automaticamente o ambiente (`IN_COLAB`):

- **Google Colab (recomendado):** basta rodar as células em ordem; quando solicitado, envie `dataset_risco_agricola_final.xlsx` (ou `.csv`) e o arquivo `.json`/`.jsonl` com a saída da simulação do ESP32.
- **Local:** instale as dependências abaixo e ajuste `CAMINHO_DATASET`/`CAMINHO_ESP32` para os caminhos locais dos arquivos.

```
scikit-learn==1.9.0
pandas
numpy
matplotlib
joblib
```

## Parte 1 — Risco ambiental

### Dados e qualidade

- **Fonte:** cruzamento SISSER (sinistro/perda agrícola) + IBGE (geolocalização por município) + Open-Meteo (clima histórico na coordenada), 9.100 linhas, específico do estado de Mato Grosso.
- **Bug de geolocalização corrigido:** nomes de município se repetem entre estados (ex.: Canarana existe em MT e na BA). O cruzamento por nome, sem filtrar por UF, pegou a latitude/longitude — e portanto a temperatura/umidade — do município errado em **817 linhas (9,0% do total)**. Essas linhas foram removidas; restaram **8.283 linhas confiáveis**.
- **Sem valores ausentes** nas colunas finais do modelo.
- **Classes desbalanceadas:** `sinistro = 0` em 92,8% das linhas, `sinistro = 1` em 7,2% — por isso os modelos usam `class_weight="balanced"` e a avaliação prioriza ROC AUC em vez de acurácia simples.

### Features

| Feature | Tipo | Equivalente no ESP32 |
|---|---|---|
| `temp_media` | numérica | `temperatura_c` |
| `umidade_media` | numérica | `umidade_pct` |
| `latitude` | numérica | `localizacao.latitude` |
| `longitude` | numérica | `localizacao.longitude` |
| `cultura` | categórica | não informada pelo sensor → usa `CULTURA_PADRAO` (moda do treino = `"Soja"`), a menos que outra seja passada |

Outras colunas do dataset original (`evento_preponderante`, `municipio_norm`, `codigo_uf`, `uf`, `pm25_media`, `pm10_media`, `ano`) foram descartadas por vazamento de dado, redundância ou falta de equivalente no ESP32.

### Preparação, treinamento e validação

- **Pré-processamento:** imputação de mediana + `StandardScaler` nas numéricas; imputação pelo mais frequente + `OneHotEncoder(handle_unknown="ignore")` na categórica.
- **Holdout:** 80% treino / 20% teste, estratificado por `sinistro`, `random_state=42`.
- **Modelos comparados** (validação cruzada, 5 folds, ROC AUC):

  | Modelo | ROC AUC médio (CV) |
  |---|---|
  | Regressão Logística | 0,660 (± 0,045) |
  | **Random Forest** | **0,750 (± 0,053)** ← escolhido |

- **Tuning (`GridSearchCV`, 5 folds, ROC AUC):** melhores hiperparâmetros `max_depth=8`, `n_estimators=500` → ROC AUC (CV) 0,751 (praticamente igual ao 0,750 sem tuning — o modelo já estava perto do ótimo para essa grade).
- **Avaliação no holdout (conjunto de teste, nunca visto):**

  | Métrica | Valor |
  |---|---|
  | Accuracy | 0,724 |
  | Precision | 0,148 |
  | Recall | 0,597 |
  | F1-score | 0,237 |
  | ROC AUC | 0,757 |

  A precisão baixa para "alto risco ambiental" (muitos falsos positivos) é esperada dado o desbalanceamento — a métrica mais confiável aqui é o ROC AUC, que não é sensível a isso.

### Interpretação

Importância por permutação (queda no ROC AUC ao embaralhar cada variável):

| Variável | Importância |
|---|---|
| `cultura` | 0,1269 |
| `longitude` | 0,0587 |
| `latitude` | 0,0391 |
| `umidade_media` | 0,0357 |
| `temp_media` | 0,0175 |

`cultura` é, disparado, a variável mais influente — mais até que as variáveis climáticas propriamente ditas.

## Parte 2 — Dataset sintético de falha mecânica

Não existe hoje nenhum dataset público real que cruze sensor ambiental com falha mecânica de máquina agrícola (bases de acidentes agrícolas não têm sensor; estudos acadêmicos usam questionário/registro de fazenda, não sensor contínuo). O mais próximo é o **AI4I 2020 Predictive Maintenance Dataset** (UCI), mas é de maquinário industrial (CNC).

Por isso, esta parte gera um **dataset sintético inspirado na estrutura do AI4I 2020**, adaptado ao contexto agrícola — os números e relações entre variáveis são simulados/ilustrativos, calibrados por raciocínio de engenharia e por evidências qualitativas da pesquisa (ex.: poeira fina penetrando vedação de rolamento em baixa umidade; vibração de desbalanceamento por acúmulo de terra, observada em estudo real com trator/colhedora no Brasil).

| Variável | O que representa | Vem do ESP32? |
|---|---|---|
| `temperatura_c` | temperatura ambiente instantânea | sim |
| `umidade_pct` | umidade relativa instantânea | sim |
| `indice_poeira` | proxy de poeira, estimado a partir da umidade | não (estimado) |
| `vibracao_magnitude` | magnitude da aceleração (9,81 m/s² = repouso) | sim (MPU6050) |
| `nivel_combustivel_pct` | nível de combustível | não |
| `eficiencia_motor_pct` | saúde/eficiência do motor | não |
| `condicao_chassi_indice` | conservação do chassi (0 a 1) | não |
| `rotacao_rodas_rpm` / `anomalia_rotacao_pct` | rotação das rodas e desvio do esperado | não |
| `horas_operacao_continua` | fadiga do equipamento na sessão | não |

O rótulo `falha_mecanica` combina versões normalizadas dessas variáveis com pesos ilustrativos, mais ruído aleatório:

| Fator | Peso |
|---|---|
| Vibração/impacto | 0,25 |
| Calor | 0,20 |
| Poeira | 0,15 |
| Eficiência do motor degradada | 0,15 |
| Condição do chassi degradada | 0,15 |
| Anomalia na rotação das rodas | 0,07 |
| Combustível crítico (<10%) | 0,03 |

Resultado: **8.000 linhas geradas, taxa de falha simulada de 12,1%**.

## Parte 3 — Risco de sinistro (modelo restrito ao firmware atual)

O modelo da Parte 2 é retreinado usando só as 4 variáveis que o firmware em uso realmente mede/deriva: `temperatura_c`, `umidade_pct`, `indice_poeira` (estimado) e `vibracao_magnitude`.

- **Modelos comparados** (CV, 5 folds, ROC AUC):

  | Modelo | ROC AUC médio (CV) |
  |---|---|
  | **Regressão Logística** | **0,797 (± 0,007)** ← escolhido |
  | Random Forest | 0,776 (± 0,010) |

- **Tuning:** melhor hiperparâmetro `C=0.1` → ROC AUC (CV) 0,797 (igual ao valor sem tuning).
- **Avaliação no holdout:**

  | Métrica | Valor |
  |---|---|
  | Accuracy | 0,708 |
  | Precision | 0,252 |
  | Recall | 0,716 |
  | F1-score | 0,373 |
  | ROC AUC | 0,783 |

- **Interpretação** (importância por permutação):

  | Variável | Importância |
  |---|---|
  | `vibracao_magnitude` | 0,1235 |
  | `temperatura_c` | 0,0962 |
  | `indice_poeira` | 0,0356 |
  | `umidade_pct` | -0,0017 (~irrelevante) |

  Vibração e temperatura dominam a decisão do modelo; a umidade isolada contribui muito pouco (o efeito dela já passa quase todo por `indice_poeira`, que é derivada dela).

O pipeline final é retreinado com 100% do dataset sintético (8.000 linhas) antes de ser usado nas previsões.

## Integração com o ESP32 (Wokwi)

- `carregar_leituras_esp32` aceita tanto uma lista JSON quanto JSON Lines (um objeto por linha, formato do Monitor Serial do Wokwi), descarta linhas inválidas e mensagens que não são leitura de sensor (sem `id_leitura`). Na simulação usada como evidência: 1 linha JSON inválida e 14 mensagens de status/evento foram descartadas, restando **11 leituras válidas**.
- **Risco ambiental:** `prever_risco_ambiental` mapeia os campos do ESP32 para as features do modelo (seção 11) e devolve `probabilidade_risco_ambiental`/`classificacao_ml` por leitura. Há uma alternativa (`prever_risco_ambiental_media_periodo`, seção 12) que agrega todas as leituras de um período e prevê uma vez só sobre a média — nessa simulação: 11 leituras agregadas, temperatura média 27,2 °C, umidade média 45,2%, probabilidade estimada 0,4330.
- **Atenção à extrapolação:** o treino cobre 21,7–27,7 °C e 60,8–80,7% de umidade; a simulação do ESP32 usada como evidência trouxe 23,1–45,2 °C e 12,5–54,0% de umidade — a faixa de umidade não tem sobreposição com o treino. Trate as probabilidades como **ranking relativo de risco entre leituras**, não como probabilidade calibrada.
- **Comparação com o alerta do firmware:** o ESP32 já calcula seu próprio `status` (NORMAL/ATENCAO/CRITICO) por regras fixas de limiar. Cruzando com a classificação do modelo nessa simulação, todas as 11 leituras (independente do `status` do firmware) saíram como "BAIXO RISCO AMBIENTAL" pelo ML — os dois sinais não precisam bater, pois medem coisas diferentes (imediato vs. padrão histórico de clima).

## Saída final recomendada

`prever_risco_sinistro_por_falha_mecanica` (seção 17) é a **saída oficial do projeto para risco de sinistro**. Ela usa exclusivamente `pipeline_mecanico_disponivel` — **não combina com o modelo de risco ambiental da Parte 1** (decisão deliberada do grupo, documentada no próprio notebook). Colunas devolvidas:

- `probabilidade_sinistro`: probabilidade contínua (0 a 1) de falha mecânica.
- `classificacao_ml`: `"RISCO DE SINISTRO"` (≥ limiar, padrão 0,5) ou `"SEM RISCO RELEVANTE"`.

Na simulação usada como evidência, 6 das 11 leituras foram classificadas como `"RISCO DE SINISTRO"`.

## Artefato exportado

`joblib.dump(pipeline_mecanico_disponivel, "modelo_risco_sinistro.joblib")` — **atenção:** o artefato exportado é só o pipeline de risco de sinistro/falha mecânica (Parte 3), não o de risco ambiental (Parte 1), coerente com a decisão de que esta é a única saída usada pelo firmware atual.

## Limitações conhecidas

- O modelo de risco ambiental usa só temperatura, umidade, latitude, longitude e cultura — distância, movimento e vibração do ESP32 não têm equivalente histórico no dataset de treino e por isso não entram nessa previsão (mas seguem disponíveis via `status` do firmware).
- Classes desbalanceadas (~7% de sinistro no dataset real): trate a probabilidade contínua como mais informativa do que o rótulo binário no limiar padrão de 0,5.
- A Parte 2/3 (risco de falha mecânica) é **inteiramente sintética/simulada** — não existe hoje um dataset público real cruzando sensor ambiental com falha mecânica de máquina agrícola.
- Há falta de sobreposição entre as faixas de treino e as leituras reais do ESP32 (seção 11) — o modelo de risco ambiental está extrapolando.

## Próximos passos

- Substituir o rótulo do SISSER por um histórico real de sinistro/manutenção de máquinas agrícolas.
- Treinar com leituras instantâneas de campo (não médias climáticas), para eliminar o problema de extrapolação.
- Reavaliar a combinação entre risco ambiental e risco de falha mecânica quando houver dado real que sustente essa relação.
