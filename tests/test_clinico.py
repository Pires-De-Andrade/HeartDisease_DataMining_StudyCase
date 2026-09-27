# -*- coding: utf-8 -*-
"""
Testes de scripts/clinico.py contra os exemplos do material complementar
("Explanation about the PPA variable") e valores conhecidos das tabelas.

Rodar: python3 -m pytest tests/  (ou python3 tests/test_clinico.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import clinico as c  # noqa: E402


# --- pressao arterial (NHBPEP 2004) -------------------------------------
def test_pa_exemplo_do_material_menina_2_anos():
    # PAS 110 entre p95 (107) e p99 (114) na coluna p75 de altura -> HAS-1 por PAS,
    # mesmo com PAD 43 normal (criterio PAS OU PAD)
    assert c.classe_pa("F", 2, 110, 43, percentil_altura=75) == "has1"


def test_pa_normal_e_pre_has():
    assert c.classe_pa("F", 2, 100, 43, 75) == "normal"      # p90 PAS = 103
    assert c.classe_pa("F", 2, 104, 43, 75) == "pre_has"     # entre p90 e p95


def test_pa_has2_pelo_p99_mais_5():
    # menino 17 anos, altura p50: p99 PAS = 143 -> has2 a partir de 148
    assert c.classe_pa("M", 17, 147, 60, 50) == "has1"
    assert c.classe_pa("M", 17, 148, 60, 50) == "has2"


def test_pa_pad_isolada_classifica():
    # menino 10 anos, altura p50: p95 PAD = 80
    assert c.classe_pa("M", 10, 100, 80, 50) == "has1"


def test_pa_adolescente_120_80_e_pre_has_mesmo_abaixo_do_p90():
    # menino 17 anos altura p95: p90 PAS = 134; 121/70 fica abaixo do p90 mas >= 120/80
    assert c.classe_pa("M", 17, 121, 70, 95) == "pre_has"


def test_pa_criterio_adulto_e_limites_de_cobertura():
    assert c.classe_pa("M", 19, 145, 85, None) == "has1"
    assert c.classe_pa("F", 18, 118, 76, None) == "normal"
    assert c.classe_pa("M", 0.5, 90, 50, 50) == "nao_avaliavel"
    assert c.classe_pa("M", 5, np.nan, 50, 50) == "nao_avaliavel"
    assert c.classe_pa("nao_informado", 5, 100, 60, 50) == "nao_avaliavel"
    assert c.classe_pa("M", 5, 100, 60, None) == "nao_avaliavel"


def test_percentil_altura_mais_proximo_escolhe_coluna_certa():
    # 72 -> coluna 75; 40 -> coluna 50 (mais proximo que 25)
    assert c.classe_pa("F", 2, 110, 43, 72) == c.classe_pa("F", 2, 110, 43, 75)


# --- IMC (CDC) ------------------------------------------------------------
def test_imc_exemplo_do_material_menina_2_anos_obesa():
    # 13 kg e 0,80 m -> IMC 20,3, acima do p95 aos 2 anos
    imc = 13 / 0.80 ** 2
    p = c.percentil_imc("F", 24, imc)
    assert p is not None and p > 95
    assert c.classe_imc(p) == "obesidade"


def test_imc_mesmo_valor_e_eutrofia_aos_15_anos():
    p = c.percentil_imc("F", 15 * 12, 20.3)
    assert c.classe_imc(p) == "eutrofia"


def test_imc_fora_da_cobertura():
    assert c.percentil_imc("F", 12, 17.0) is None
    assert c.classe_imc(None) == "nao_avaliavel"


# --- altura -----------------------------------------------------------------
def test_percentil_altura_mediana_cdc():
    # mediana CDC aos 24 meses (meninas) ~ 85-86 cm; aos 10 anos (meninos) ~ 138 cm
    assert 35 < c.percentil_altura("F", 24, 85.5) < 65
    assert 35 < c.percentil_altura("M", 120, 138) < 65
    assert c.percentil_altura("M", 6, 67) is not None          # tabela de comprimento
    assert c.percentil_altura("M", 120, np.nan) is None


# --- FC e faixa etaria ------------------------------------------------------
def test_fc():
    assert c.classe_fc(0.1, 150) == "normal"
    assert c.classe_fc(15, 130) == "taquicardia"
    assert c.classe_fc(8, 60) == "bradicardia"
    assert c.classe_fc(8, np.nan) == "nao_avaliavel"


def test_faixa_etaria():
    assert [c.faixa_etaria(x) for x in (0.5, 2, 5.9, 6, 11.9, 12, 19)] == \
        ["lactente", "pre_escolar", "pre_escolar", "escolar", "escolar", "adolescente", "adolescente"]
    assert c.faixa_etaria(np.nan) is None


if __name__ == "__main__":
    testes = [v for k, v in globals().items() if k.startswith("test_")]
    for t in testes:
        t()
        print(f"  [ok] {t.__name__}")
    print(f"{len(testes)} testes passaram")
