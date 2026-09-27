# -*- coding: utf-8 -*-
"""
Definicoes compartilhadas pelo Modulo 2 (diagnostico de qualidade e limpeza).

Tudo o que decide "o que e invalido" mora aqui, para que o diagnostico
(02_diagnostico_qualidade.py) e a limpeza (02_limpeza.py) usem exatamente
as mesmas regras e as contagens de um batam com as do outro.

FAIXAS DE PLAUSIBILIDADE
------------------------
Sao as mesmas janelas usadas no 01_analise_univariada.py (dicionario QUANT);
o diagnostico confere suas contagens contra output/tabelas/resumo_univariado.csv,
o que acusa qualquer divergencia entre os dois modulos.
Sao limites de PLAUSIBILIDADE fisiologica, nao de normalidade: um valor
anormal mas possivel (ex.: PA de hipertenso) e informacao sobre o alvo e
deve ser mantido. Valores altos extremos de peso, altura e IMC em criancas
costumam ser reais (Freedman et al., Am J Clin Nutr 2015;102:1000-6), por
isso so o implausivel e anulado.

As faixas fixas sao a primeira barreira (valores impossiveis em qualquer
idade). Peso, altura e IMC passam tambem por uma segunda barreira, POR IDADE
E SEXO: o escore-z das curvas da OMS, com os limites de implausibilidade do
WHO Anthro (peso/idade < -6 ou > +5; altura/idade < -6 ou > +6;
IMC/idade < -5 ou > +5). Tabelas: WHO Child Growth Standards 2006 (0-5 anos)
e WHO Growth Reference 2007 (5-19 anos), em data/referencias/oms/.
"""

import os
import re
import unicodedata
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_RAW = os.path.join(BASE, "UCMF_raw.csv")
OUT_TAB = os.path.join(BASE, "output", "tabelas")

FAIXAS = {
    #  variavel         : (min, max) inclusivos
    "Peso":           (0.5, 150),     # kg
    "Altura":         (30, 200),      # cm
    "IMC":            (8, 45),        # kg/m²
    "IDADE":          (0, 19),        # anos
    "PA SISTOLICA":   (50, 200),      # mmHg
    "PA DIASTOLICA":  (20, 150),      # mmHg
    "FC":             (40, 220),      # bpm
}

# Populacao do estudo: 0 a 19 anos (PDF "Variaveis do conjunto de dados").
# Idade em anos completos: 19,9 anos ainda e "19 anos".
IDADE_LIMITE_POPULACAO = 20

# Ultimo ano plausivel de coleta (mesmo criterio do 01_analise_univariada.py).
ANO_LIMITE = 2010

# Origem do calendario de datas seriais do Excel (Windows).
EXCEL_ORIGEM = pd.Timestamp("1899-12-30")

# Nomes finais em snake_case (contrato de dados, docs/contrato-dados.md).
SNAKE = {
    "ID": "id", "Peso": "peso", "Altura": "altura", "IMC": "imc",
    "Atendimento": "atendimento", "DN": "dn", "IDADE": "idade",
    "Convenio": "convenio", "PULSOS": "pulsos",
    "PA SISTOLICA": "pa_sistolica", "PA DIASTOLICA": "pa_diastolica",
    "PPA": "ppa", "NORMAL X ANORMAL": "normal_x_anormal", "B2": "b2",
    "SOPRO": "sopro", "FC": "fc", "HDA 1": "hda1", "HDA2": "hda2",
    "SEXO": "sexo", "MOTIVO1": "motivo1", "MOTIVO2": "motivo2",
}

# Rotulos que nao sao variacao de grafia e por isso exigem mapa explicito.
MAPA_SEXO = {"M": "M", "Masculino": "M", "masculino": "M",
             "F": "F", "Feminino": "F",
             "Indeterminado": np.nan}           # "nao informado", nao e categoria
MAPA_ALVO = {"Normal": "Normal", "Normais": "Normal",
             "Anormal": "Anormal", "anormal": "Anormal"}
PPA_SEM_CLASSE = ["#VALUE!", "Não Calculado"]    # erro de formula / PA ausente

# Categoricas cujas variantes sao so caixa, acento ou pontuacao.
CATEG_GRAFIA = ["PULSOS", "B2", "SOPRO", "HDA 1", "HDA2",
                "MOTIVO1", "MOTIVO2", "Convenio"]

RE_INTERVALO = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:-|a)\s*(\d+(?:[.,]\d+)?)\s*$")


def carregar_bruto():
    """Le a base bruta toda como texto, como no Modulo 1."""
    return pd.read_csv(CSV_RAW, dtype=str, keep_default_na=False, na_values=[""])


def colunas_conteudo(df):
    """Todas as colunas exceto o identificador (usado para achar duplicatas)."""
    return [c for c in df.columns if c.upper() != "ID"]


# -------------------------------------------------------------------------
# Datas
# -------------------------------------------------------------------------
def parse_data(s):
    """
    Converte uma coluna de datas brutas.

    Aceita dd/mm/aa e numeros seriais do Excel (ex.: '38574' = 10/08/2005).
    O ano com 2 digitos segue a janela do strptime: 69-99 -> 19xx, 00-68 -> 20xx.
    Retorna (datas, eh_serial).
    """
    d = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    serial = pd.to_numeric(s, errors="coerce")
    eh_serial = d.isna() & serial.notna()
    d_serial = EXCEL_ORIGEM + pd.to_timedelta(serial.where(eh_serial), unit="D")
    return d.fillna(d_serial), eh_serial


def corrigir_seculo(d):
    """
    Data de nascimento no futuro so pode ser seculo trocado pela janela do
    strptime ('55' lido como 2055): recua 100 anos. Retorna (datas, corrigidas).
    """
    futuro = d.dt.year > ANO_LIMITE
    return d.where(~futuro, d - pd.DateOffset(years=100)), futuro


def idade_por_datas(atendimento, dn):
    """Idade em anos no dia do atendimento."""
    return (atendimento - dn).dt.days / 365.25


# -------------------------------------------------------------------------
# Numericas
# -------------------------------------------------------------------------
def fora_faixa(x, col):
    lo, hi = FAIXAS[col]
    return x.notna() & ((x < lo) | (x > hi))


def fc_intervalo(s):
    """Ponto medio de FC digitada como intervalo ('80-90', '156 a 168')."""
    m = s.str.extract(RE_INTERVALO)
    a = pd.to_numeric(m[0].str.replace(",", "."), errors="coerce")
    b = pd.to_numeric(m[1].str.replace(",", "."), errors="coerce")
    return (a + b) / 2


def imc_calculado(peso, altura_cm):
    return peso / (altura_cm / 100) ** 2


# -------------------------------------------------------------------------
# Curvas da OMS - escore-z por idade e sexo
# -------------------------------------------------------------------------
DIR_OMS = os.path.join(BASE, "data", "referencias", "oms")
DIAS_POR_MES = 30.4375

# indicador -> arquivos (0-5 anos por dia; 5-19 anos por mes). Peso/idade da
# referencia 2007 so vai ate 10 anos: acima disso o peso e checado pelo IMC.
TABELAS_OMS = {
    "peso":   ["wfa_{s}_0-5.xlsx", "wfa_{s}_5-10.xlsx"],
    "altura": ["lhfa_{s}_0-5.xlsx", "hfa_{s}_5-19.xlsx"],
    "imc":    ["bfa_{s}_0-5.xlsx", "bfa_{s}_5-19.xlsx"],
}
SEXO_OMS = {"M": "boys", "F": "girls"}

# Limites de implausibilidade do WHO Anthro / AnthroPlus.
LIMITES_Z = {"peso": (-6, 5), "altura": (-6, 6), "imc": (-5, 5)}

_LMS = {}


def lms_oms(indicador, sexo):
    """Tabela L, M, S por idade em dias (0 a 19 anos), das planilhas da OMS."""
    chave = (indicador, sexo)
    if chave not in _LMS:
        partes = []
        for arq in TABELAS_OMS[indicador]:
            t = pd.read_excel(os.path.join(DIR_OMS, arq.format(s=SEXO_OMS[sexo])))
            dias = t["Day"] if "Day" in t else t["Month"] * DIAS_POR_MES
            partes.append(pd.DataFrame({"dias": dias, "L": t["L"], "M": t["M"], "S": t["S"]}))
        _LMS[chave] = pd.concat(partes).sort_values("dias").reset_index(drop=True)
    return _LMS[chave]


def escore_z(valor, idade_anos, sexo, indicador):
    """
    Escore-z da OMS. Fora da cobertura da tabela (idade ausente, acima de
    19 anos ou peso acima de 10 anos) retorna NaN.
    Peso e IMC usam o ajuste de cauda da OMS acima de |z| = 3 (a distancia
    entre 2 e 3 DP e estendida linearmente), como no WHO Anthro.
    """
    lms = lms_oms(indicador, sexo)
    dias = idade_anos * 365.25
    cobre = dias.between(lms["dias"].min(), lms["dias"].max())
    L = pd.Series(np.interp(dias, lms["dias"], lms["L"]), index=valor.index).where(cobre)
    M = pd.Series(np.interp(dias, lms["dias"], lms["M"]), index=valor.index).where(cobre)
    S = pd.Series(np.interp(dias, lms["dias"], lms["S"]), index=valor.index).where(cobre)
    z = ((valor / M) ** L - 1) / (L * S)
    if indicador == "altura":
        return z

    def sd(k):
        return M * (1 + L * S * k) ** (1 / L)

    alto = z > 3
    baixo = z < -3
    z = z.mask(alto, 3 + (valor - sd(3)) / (sd(3) - sd(2)))
    z = z.mask(baixo, -3 + (valor - sd(-3)) / (sd(-2) - sd(-3)))
    return z


NOME_Z = {"peso": "peso/idade", "altura": "altura/idade", "imc": "IMC/idade"}


def descrever_limite_z(indicador):
    lo, hi = LIMITES_Z[indicador]
    return f"OMS: z de {NOME_Z[indicador]} < {lo} ou > +{hi}".replace("-", "−")


def validar_tabelas_oms(tolerancia=0.01):
    """
    Confere a implementacao do escore-z contra as proprias planilhas da OMS:
    as colunas -3 a +3 DP tem de resultar em z = -3 ... +3.
    """
    colunas = {-3: "SD3neg", -2: "SD2neg", -1: "SD1neg", 0: "M",
               1: "SD1", 2: "SD2", 3: "SD3"}
    pior = 0.0
    for ind, arqs in TABELAS_OMS.items():
        for s, nome in SEXO_OMS.items():
            for arq in arqs:
                t = pd.read_excel(os.path.join(DIR_OMS, arq.format(s=nome)))
                dias = t["Day"] if "Day" in t else t["Month"] * DIAS_POR_MES
                for k, col in colunas.items():
                    z = escore_z(t[col].astype(float), dias / 365.25, s, ind)
                    pior = max(pior, float((z - k).abs().max()))
    assert pior < tolerancia, f"escore-z diverge das tabelas da OMS ({pior:.4f})"
    return pior


# Abaixo de 1 ano so o limite superior vale: as curvas da OMS sao de bebes a
# termo, e um prematuro pequeno para a idade cronologica e plausivel.
IDADE_SO_LIMITE_SUPERIOR = 1.0


def z_conservador(valor, idade_anos, sexo, indicador):
    """
    Escore-z pelo sexo do registro. Sexo ausente: o z mais proximo de zero
    entre os dois sexos, para so marcar o que e implausivel em ambos.
    """
    zm = escore_z(valor, idade_anos, "M", indicador)
    zf = escore_z(valor, idade_anos, "F", indicador)
    neutro = zm.where(zm.abs() <= zf.abs(), zf)
    return zm.where(sexo == "M", zf.where(sexo == "F", neutro))


def fora_limite_z(z, idade_anos, indicador):
    lo, hi = LIMITES_Z[indicador]
    baixo = (z < lo) & (idade_anos >= IDADE_SO_LIMITE_SUPERIOR)
    return baixo | (z > hi)


def avaliar_antropometria(peso, altura, idade_anos, sexo):
    """
    Aplica os limites de implausibilidade da OMS a peso, altura e IMC.

    Se peso E altura sao implausiveis no mesmo sentido mas o IMC entre eles e
    plausivel, as medidas sao coerentes entre si e o que nao bate e a idade
    (erro nas datas): marca a idade, nao as medidas.

    Retorna mascaras: idade_incompativel, peso, altura, imc e o IMC calculado
    com as medidas que sobraram.
    """
    zp = z_conservador(peso, idade_anos, sexo, "peso")
    za = z_conservador(altura, idade_anos, sexo, "altura")
    bp = fora_limite_z(zp, idade_anos, "peso")
    ba = fora_limite_z(za, idade_anos, "altura")

    imc_par = imc_calculado(peso, altura)
    zi_par = z_conservador(imc_par, idade_anos, sexo, "imc")
    idade_incomp = bp & ba & (np.sign(zp) == np.sign(za)) & \
        zi_par.between(*LIMITES_Z["imc"])
    bp, ba = bp & ~idade_incomp, ba & ~idade_incomp

    idade_ok = idade_anos.mask(idade_incomp)
    imc = imc_calculado(peso.mask(bp), altura.mask(ba))
    bi = fora_limite_z(z_conservador(imc, idade_ok, sexo, "imc"), idade_ok, "imc")
    return {"idade_incompativel": idade_incomp, "peso": bp, "altura": ba,
            "imc": bi, "imc_valor": imc}


# -------------------------------------------------------------------------
# Categoricas
# -------------------------------------------------------------------------
def chave_grafia(v):
    """Chave que ignora caixa, acento, espacos extras e ponto final."""
    if pd.isna(v):
        return v
    v = unicodedata.normalize("NFKD", v)
    v = "".join(ch for ch in v if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", v).strip().rstrip(".").strip().lower()


def unificar_grafia(s):
    """
    Troca cada variante pela grafia mais frequente do seu grupo.
    Empates sao resolvidos pela ordem alfabetica, para o resultado nao
    depender da versao do pandas.
    Retorna (serie unificada, tabela variante -> canonica).
    """
    chave = s.map(chave_grafia)
    freq = (pd.DataFrame({"chave": chave, "valor": s}).dropna()
            .value_counts().reset_index(name="n")
            .sort_values(["chave", "n", "valor"], ascending=[True, False, True]))
    canon = freq.drop_duplicates("chave").set_index("chave")["valor"]
    mapa = freq.assign(canonica=freq["chave"].map(canon))
    return chave.map(canon), mapa[["valor", "canonica", "n"]]
