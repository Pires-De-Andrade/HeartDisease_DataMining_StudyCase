# -*- coding: utf-8 -*-
"""
Classificacoes clinicas pediatricas usadas na construcao de atributos (Modulo 4).

Todas as funcoes sao deterministicas: dependem apenas das tabelas de referencia
em scripts/referencias/ (fontes no README de la), nunca de estatisticas da
amostra. Por isso podem ser aplicadas a base inteira sem vazamento treino/teste.

Convencao: "nao_avaliavel" quando falta insumo (valor ausente, idade fora da
cobertura da tabela, sexo nao informado).
"""

import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "referencias")
SEXO_CDC = {"M": 1, "F": 2}
COLUNAS_ALTURA = np.array([5, 10, 25, 50, 75, 90, 95])
NAO_AVALIAVEL = "nao_avaliavel"

ORDEM_IMC = ["baixo_peso", "eutrofia", "excesso_peso", "obesidade"]
ORDEM_PA = ["normal", "pre_has", "has1", "has2"]
ORDEM_FC = ["bradicardia", "normal", "taquicardia"]
ORDEM_FAIXA = ["lactente", "pre_escolar", "escolar", "adolescente"]


@lru_cache(maxsize=None)
def _tabela(nome):
    return pd.read_csv(os.path.join(REF, nome))


@lru_cache(maxsize=None)
def _curvas_lms(nome, sexo):
    """Arrays (idade_meses, L, M, S) de um sexo, para interpolacao rapida."""
    t = _tabela(nome)
    t = t[t["Sex"] == SEXO_CDC[sexo]].sort_values("Agemos")
    return tuple(t[c].to_numpy(float) for c in ("Agemos", "L", "M", "S"))


def _lms(nome, sexo, idade_meses):
    """Interpola L, M, S para (sexo, idade em meses); None fora da cobertura."""
    if sexo not in SEXO_CDC or pd.isna(idade_meses):
        return None
    meses, L, M, S = _curvas_lms(nome, sexo)
    if not (meses[0] <= idade_meses <= meses[-1]):
        return None
    return tuple(np.interp(idade_meses, meses, c) for c in (L, M, S))


def percentil_lms(valor, lms):
    """Percentil de `valor` na curva LMS (metodo de Cole, usado pelo CDC)."""
    L, M, S = lms
    z = math.log(valor / M) / S if abs(L) < 1e-9 else ((valor / M) ** L - 1) / (L * S)
    return 100 * 0.5 * (1 + math.erf(z / math.sqrt(2)))


# -------------------------------------------------------------------------
# percentis antropometricos
# -------------------------------------------------------------------------
def percentil_altura(sexo, idade_meses, altura_cm):
    """Percentil de altura para idade e sexo (comprimento < 24 m, estatura >= 24 m)."""
    if pd.isna(altura_cm) or altura_cm <= 0 or pd.isna(idade_meses):
        return None
    nome = "cdc_comprimento_0a36m_lms.csv" if idade_meses < 24 else "cdc_estatura_lms.csv"
    lms = _lms(nome, sexo, idade_meses)
    return None if lms is None else percentil_lms(altura_cm, lms)


def percentil_imc(sexo, idade_meses, imc):
    """Percentil de IMC para idade e sexo; None abaixo de 24 meses."""
    if pd.isna(imc) or imc <= 0:
        return None
    lms = _lms("cdc_imc_lms.csv", sexo, idade_meses)
    return None if lms is None else percentil_lms(imc, lms)


# -------------------------------------------------------------------------
# classes clinicas
# -------------------------------------------------------------------------
def classe_imc(percentil):
    if percentil is None:
        return NAO_AVALIAVEL
    if percentil < 5:
        return "baixo_peso"
    if percentil < 85:
        return "eutrofia"
    if percentil < 95:
        return "excesso_peso"
    return "obesidade"


@lru_cache(maxsize=None)
def _limiares_pa(sexo, idade, coluna_altura):
    """(p90, p95, p99) como pares (PAS, PAD) para sexo, idade inteira e coluna de altura."""
    t = _tabela("nhbpep_2004_pa.csv")
    t = t[(t.sexo == sexo) & (t.idade_anos == idade) & (t.percentil_altura == coluna_altura)]
    t = t.set_index("percentil_pa")
    return tuple((int(t.pas[p]), int(t.pad[p])) for p in (90, 95, 99))


def classe_pa(sexo, idade_anos, pas, pad, percentil_altura=None):
    """
    Classificacao NHBPEP 2004 (criterio PAS OU PAD):
      normal < p90 | pre_has p90-p95 (ou >= 120/80 a partir de 12 anos)
      has1 p95 ate p99+5 | has2 >= p99+5.
    >= 18 anos: criterio adulto (120/80, 140/90, 160/100). < 1 ano: nao avaliavel.
    """
    if pd.isna(pas) or pd.isna(pad) or pd.isna(idade_anos):
        return NAO_AVALIAVEL
    idade = int(idade_anos)
    if idade < 1:
        return NAO_AVALIAVEL
    if idade >= 18:
        p90, p95, p99 = (120, 80), (140, 90), (160, 100)
    else:
        if sexo not in SEXO_CDC or percentil_altura is None:
            return NAO_AVALIAVEL
        coluna = int(COLUNAS_ALTURA[np.abs(COLUNAS_ALTURA - percentil_altura).argmin()])
        p90, p95, (s99, d99) = _limiares_pa(sexo, idade, coluna)
        p99 = (s99 + 5, d99 + 5)
    if pas >= p99[0] or pad >= p99[1]:
        return "has2"
    if pas >= p95[0] or pad >= p95[1]:
        return "has1"
    if pas >= p90[0] or pad >= p90[1] or (idade >= 12 and (pas >= 120 or pad >= 80)):
        return "pre_has"
    return "normal"


def classe_fc(idade_anos, fc):
    if pd.isna(fc) or pd.isna(idade_anos):
        return NAO_AVALIAVEL
    t = _tabela("fc_faixas.csv")
    faixa = t[(t.idade_min_anos <= idade_anos) & (idade_anos < t.idade_max_anos)]
    if faixa.empty:
        return NAO_AVALIAVEL
    lo, hi = faixa.iloc[0][["fc_min", "fc_max"]]
    return "bradicardia" if fc < lo else "taquicardia" if fc > hi else "normal"


def faixa_etaria(idade_anos):
    if pd.isna(idade_anos):
        return None
    if idade_anos < 2:
        return "lactente"
    if idade_anos < 6:
        return "pre_escolar"
    if idade_anos < 12:
        return "escolar"
    return "adolescente"
