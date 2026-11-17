# Triagem de cardiopatia pediátrica antes da ausculta

Trabalho da disciplina de Mineração de Dados (Instituto de Informática, UFG), Grupo 1:
João Guilherme Pires de Andrade, Gustavo Henrique Barros, Gustavo Almeida Leite e
Felipe Brito Albernaz.

**Pergunta.** Quanto do diagnóstico de patologia cardíaca em crianças e adolescentes pode ser
antecipado com as informações disponíveis *antes* da ausculta do especialista? Esse desempenho
se mantém ao longo do tempo?

**Dados.** A base UCMF, do Real Hospital Português (Recife): 12.873 atendimentos de 0 a 19 anos,
entre 2003 e 2010. Depois da limpeza e da exclusão dos encaminhados com cardiopatia já
estabelecida, a população de estudo tem 10.311 registros, com 39,6% de Anormais.

## Principais resultados (conjunto de teste)

Os valores são da AUC-ROC do *gradient boosting*, com IC 95% por bootstrap:

| Informação disponível | Split aleatório | Split temporal (treino ≤ 2008) |
|---|---|---|
| Só idade (regressão logística) | 0,63 | 0,63 |
| E0: encaminhamento | 0,75 | 0,73 |
| E1: + exame sem estetoscópio | 0,78 [0,76; 0,80] | 0,77 [0,75; 0,78] |
| E2: + ausculta (teto; vazamento) | 0,95 | 0,95 |

No ponto de corte de triagem, priorizar 60% da fila captura 81% das cardiopatias. As hipóteses e
o protocolo foram registrados antes do teste, em [docs/proposta.md](docs/proposta.md). Os desvios
do protocolo estão na seção 8 desse documento.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `UCMF_raw.csv` | Base original |
| `data/processed/` | Base limpa (`UCMF_clean.csv`) e atributos clínicos (`UCMF_features.csv`) |
| `data/referencias/oms/`, `scripts/referencias/` | Tabelas clínicas: OMS, CDC, NHBPEP e faixas de FC |
| `scripts/` | Pipeline numerada (abaixo) e módulos auxiliares (`clinico.py`, `transformadores.py`, ...) |
| `output/` | Tabelas e figuras geradas. Os resultados do artigo estão em `output/modelagem/` |
| `paper/` | Artigo em LaTeX (IEEEtran). Veja `paper/README.md` |
| `docs/proposta.md` | Pergunta, hipóteses e protocolo pré-registrados |
| `tests/` | Testes das funções clínicas |

## Como reproduzir

Requer Python 3.11 ou mais recente. Todos os comandos são executados na raiz do repositório.

```bash
pip install -r requirements.txt

# Análise exploratória e pré-processamento (A1)
python scripts/01_analise_univariada.py
python scripts/02_diagnostico_qualidade.py
python scripts/02_limpeza.py            # -> data/processed/UCMF_clean.csv
python scripts/04_transformacoes.py     # -> data/processed/UCMF_features.csv
python scripts/03_analise_bivariada.py

# Modelagem (A2): duas fases, como no protocolo
python scripts/08_modelagem.py --fase cv      # só o treino: grade e pontos de corte
python scripts/08_modelagem.py --fase teste   # avalia no teste uma única vez
python scripts/07_figuras_artigo.py           # -> paper/figures/
```

Observações:

- A fase `teste` se recusa a rodar se `scripts/08_modelagem.py` tiver alterações não
  commitadas. Ela também confere se os hiperparâmetros e os pontos de corte coincidem com os
  fixados na fase `cv` (`output/modelagem/cv_fixados.csv`).
- Cada fase leva cerca de 15 minutos em 4 núcleos.
- Todas as etapas aleatórias usam `random_state = 42`.
- As tabelas da OMS já estão no repositório. `scripts/00_baixar_referencias_oms.py` só é
  necessário para baixá-las de novo.
- `scripts/05_figuras_relatorio.py` e `06_relatorio_docx.py` geram o relatório da A1. O segundo
  requer `python-docx`.
- Testes: `python -m pytest tests`.

## Uso de IA generativa

O código e a primeira versão do texto foram escritos com o apoio do Claude (Anthropic), por meio
do Claude Code. A pergunta, as hipóteses, as decisões metodológicas e a revisão foram do grupo.
Os detalhes estão na seção final do artigo.
