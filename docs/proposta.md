# Proposta de pesquisa — A2

**Registrada em:** 12/11/2026, antes da execução de qualquer modelo.
**Grupo 1 — Mineração de Dados, Bacharelado em IA, UFG.**

Este documento fixa a pergunta, as hipóteses e o protocolo experimental antes de qualquer
treinamento. Toda alteração posterior deve ser registrada na seção [Desvios do protocolo](#8-desvios-do-protocolo),
com data e justificativa, e entra no artigo como tal.

---

## 1. Problema

Grande parte das crianças encaminhadas à cardiologia pediátrica não tem cardiopatia. Em geral,
o encaminhamento se deve a um sopro inocente. Ao mesmo tempo, há poucos especialistas, e
o diagnóstico tardio de cardiopatia congênita é mais frequente onde o acesso é limitado, como no
Nordeste brasileiro. Priorizar a fila de encaminhamentos com as informações disponíveis
*antes* da ausculta do especialista é, portanto, um problema clínico real.

A base UCMF vem desse contexto: são atendimentos de um serviço de cardiologia pediátrica do
Real Hospital Português (Recife, PE). Ela é uma **fatia** desse mundo, com limites próprios:
- **Pacientes:** só crianças já encaminhadas.
- **Local:** um único centro.
- **Período:** 2003–2010.
- **Acesso:** maioria de convênios.
- **Alvo:** Normal/Anormal, sem o tipo de cardiopatia.

## 2. Trabalhos anteriores com a mesma base

| Trabalho | População | Ausculta nos atributos | Validação | Resultado |
|---|---|---|---|---|
| Ferreira et al., 2012 [1] | 169 crianças (2011) | com e sem sopro | VC 10×10 | sem sopro: acurácia 79,3%, sensibilidade 0,28 (ZeroR 76,3%) |
| Ferreira et al., 2013 [2] | 7.199, 2–19 anos | com e sem sopro (o "sem" mantém B2 e o motivo "presença de sopros") | VC 10×10 | sem sopro: acurácia 73,1%, sensibilidade 0,37 (ZeroR 65,2%) |
| Serra Neto, 2020 [3] | 9.484, 2–19 anos (53% da base; remove quem tem mais de 10 ausentes) | sopro, B2 e o atributo "HEL", derivado do alvo | um único split 67/33 | cerca de 92,7% de acurácia balanceada |
| Serra Neto et al., 2022 [4] | a mesma de [3] | a mesma de [3] | VC estratificada + teste | cerca de 92,5% de acurácia balanceada |

**Lacuna.** Nenhum desses trabalhos fez as seguintes coisas:
1. separar os atributos pelo momento do atendimento em que estão disponíveis;
2. incluir menores de 2 anos;
3. avaliar com métricas de triagem (AUC, sensibilidade num ponto de corte fixado, calibração);
4. comparar com um modelo de referência clínico trivial;
5. fazer validação temporal.

Os trabalhos [1, 2] concluem que, sem o sopro, os demais atributos pouco contribuem. Este
trabalho reavalia essa conclusão com um protocolo que corrige os pontos acima.

**Procedência dos dados.** [3, 4] reportam 17.874 pacientes brutos, mas as tabelas descritivas
são idênticas às de [2], que tem 7.603 brutos. A nossa base tem 12.873 registros brutos. Não há
como garantir que as extrações sejam as mesmas, e por isso as comparações numéricas com a literatura são
aproximadas.

## 3. Pergunta de pesquisa

> Na triagem de encaminhamentos à cardiologia pediátrica, quanto do diagnóstico de cardiopatia
> pode ser antecipado com as informações disponíveis antes da ausculta, e esse desempenho se
> mantém ao longo do tempo?

## 4. Hipóteses

| | Hipótese | Teste |
|---|---|---|
| **H1, etapas** | O desempenho cresce a cada etapa de informação: E1 > E0 > modelo só com idade. | Diferença de AUC-ROC no teste, com IC 95% por bootstrap; a hipótese se sustenta se o IC excluir zero. |
| **H2, replicação** | Na subpopulação de 2 a 19 anos, com especificidade fixada em cerca de 0,92, a sensibilidade em E1 fica próxima da faixa reportada por [1, 2] (0,28–0,37). | Sensibilidade, com IC 95%, no ponto de corte que dá especificidade ≈ 0,92 no treino. |
| **H3, lactentes** | Sem ausculta (E1), o desempenho em menores de 2 anos é maior que na faixa de 2 a 19 anos. | AUC-ROC estratificada por faixa (< 2 contra 2–19), com IC 95%. |
| **H4, tempo** | O desempenho cai no split temporal em relação ao split aleatório. | Diferença de AUC-ROC e de Brier entre os dois splits, com IC 95%. |

**Verificação (não é hipótese):** E2, com a ausculta, deve se aproximar dos resultados de [2–4]
(cerca de 0,9). Isso confirma que o desempenho reportado na literatura vem da ausculta.

Uma hipótese refutada é resultado e entra no artigo.

## 5. Escopo

**Tarefa:** classificação binária, `normal_x_anormal` (Anormal = 1).

**População principal:** base limpa da A1 (11.401 registros), excluindo
`MOTIVO1 = "1 - Cardiopatia já estabelecida"` (1.090 registros). São pacientes já diagnosticados,
fora do problema de triagem. Restam **10.311 registros, 39,6% Anormal**.

| Subpopulação | n | % Anormal | Uso |
|---|---|---|---|
| < 2 anos | 3.092 | 52,0 | H3 |
| 2–19 anos | 6.534 | 33,4 | H2 e H3 (comparável a [2]: 34,8%) |
| idade ausente | 685 | — | entra no treino como "não avaliável"; fica fora da estratificação de H2 e H3 |

**Etapas de informação**, todas cumulativas:

| Etapa | Momento do atendimento | Atributos |
|---|---|---|
| **E0, encaminhamento** | antes da consulta | idade (e faixa etária), sexo, MOTIVO1, MOTIVO2, HDA1 |
| **E1, exame sem estetoscópio** | triagem e exame inicial | E0 + peso, altura, FC, classe de IMC, classe de PA, classe de FC, pulsos |
| **E2, ausculta (teto)** | cardiologista | E1 + sopro, B2 |

Decisões sobre os atributos:
- `MOTIVO2 = "6 - Sopro"` (2.247 registros) fica em E0. É o sopro relatado por quem encaminhou,
  informação legítima de antes da consulta. Uma **ablação** de E0 e E1 sem esse motivo mede quanto
  o modelo depende dele.
- Pulsos ficam em E1, porque são palpados, não auscultados.
- Continuam fora: ID, convênio, HDA2, PPA original, IMC contínuo e datas, como já decidido na A1.
  A data de atendimento é usada **só para definir o split temporal**, nunca como atributo.

**Fora do escopo:** tipo de cardiopatia; sinal de áudio; metaheurísticas de seleção de
atributos; deep learning tabular; validação externa na base CirCor DigiScope [5] (trabalho futuro).

## 6. Protocolo experimental

**Pré-processamento:** o pipeline da A1 (`scripts/transformadores.py`). Imputação, codificação e
padronização são ajustadas **só no treino** de cada split, e o mesmo vale dentro de cada fold da validação
cruzada interna.

**Splits:**
- **Aleatório:** 80/20 estratificado pelo alvo, `random_state = 42`, refeito sobre a população
  principal. O `split_indices.csv` da A1 foi gerado sobre os 11.401 registros e não é reaproveitado.
- **Temporal:** treino com atendimentos de 2003 a 2008 (7.520 registros, 42,3% Anormal) e teste
  com 2009–2010 (2.715 registros, 32,3% Anormal). Ficam de fora 76 registros sem data de atendimento,
  e os anteriores a 2003 vão para o treino.

**Modelos:**

| Papel | Modelo |
|---|---|
| Referência trivial | Prevalência (DummyClassifier) |
| Referência clínica | Regressão logística só com a idade |
| Modelo simples | Regressão logística (L2, `class_weight="balanced"`) |
| Modelo mais complexo | Gradient boosting (`HistGradientBoostingClassifier`) |

**Ajuste de hiperparâmetros:** grade pequena, com validação cruzada estratificada de 5 folds no treino
e AUC-ROC como critério. A grade de cada modelo fica no código e não é alterada depois que o
teste for visto.

**Métricas** (no teste, com IC 95% por bootstrap de 1.000 reamostragens):
- **principal:** AUC-ROC;
- **secundárias:** AUC-PR, sensibilidade e especificidade no ponto de corte fixado no treino, e Brier.

**Ponto de corte:** definido no treino por validação cruzada. Há dois cortes: sensibilidade ≥ 0,80 (cenário de
triagem) e especificidade ≈ 0,92 (para a comparação de H2).

**Interpretação:** matriz de confusão do melhor modelo em E1; importância por permutação
no teste; análise dos erros por faixa etária.

**Reprodutibilidade:** `random_state = 42` em todos os componentes; versões em
`requirements.txt`; o teste é avaliado **uma única vez** por configuração, depois de fixados os modelos.

## 7. Limitações conhecidas antes do experimento

- A análise exploratória da A1 usou a base inteira, incluindo o que vira conjunto de teste.
  As decisões de limpeza e transformação foram tomadas com essa visão.
- Um único centro, apenas pacientes encaminhados, período de 2003 a 2010: a generalização para outras
  populações não é avaliada.
- O alvo é binário, sem o tipo de cardiopatia. O rótulo pode ter sido definido com a ausculta
  e o ecocardiograma, que não estão nos dados.
- Não há como confirmar que a nossa extração é a mesma usada por [2–4].

## 8. Desvios do protocolo

*(Nenhum até o momento. Registrar aqui: data, o que mudou, por quê, e se foi antes ou depois de
ver o teste.)*

---

## Referências

[1] P. Ferreira et al., "Detecting cardiac pathologies from annotated auscultations," in *Proc. IEEE
CBMS*, 2012. DOI: 10.1109/CBMS.2012.6266358

[2] P. Ferreira et al., "Knowledge on heart condition of children based on demographic and
physiological features," in *Proc. IEEE CBMS*, 2013, pp. 314–319. DOI: 10.1109/CBMS.2013.6627808

[3] M. T. R. Serra Neto, "Feature selection with the MS-EPSO algorithm to predict cardiac pathology
in children and teenagers," dissertação de mestrado, FCUP, Universidade do Porto, 2020.
DOI: 10.34626/hb8h-r711

[4] M. T. R. Serra Neto, I. Dutra, and M. A. F. Mollinetti, "Map-Optimize-Learn: Predicting cardiac
pathology in children and teenagers with a deep learning based tabular learning method," in *Proc.
IJCNN*, 2022. DOI: 10.1109/IJCNN55064.2022.9889788

[5] J. Oliveira et al., "The CirCor DigiScope dataset: From murmur detection to murmur
classification," *IEEE J. Biomed. Health Inform.*, vol. 26, no. 6, pp. 2524–2535, 2022.

Referências de contexto (motivação, a verificar na fonte antes de entrar no artigo):
- S. G. Finlayson et al., "The clinician and dataset shift in artificial intelligence," *N. Engl. J. Med.*,
  vol. 385, no. 3, pp. 283–286, 2021. DOI: 10.1056/NEJMc2104626
- D. Agniel, I. S. Kohane, and G. M. Weber, "Biases in electronic health record data due to processes
  within the healthcare system," *BMJ*, vol. 361, k1479, 2018.
- Estudos sobre a proporção de sopros inocentes entre os encaminhamentos e sobre o acesso à cardiologia pediátrica
  no Nordeste: levantados em 12/11/2026, ainda sem confirmação na fonte.
