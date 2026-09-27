# -*- coding: utf-8 -*-
"""
MODULO 2a - DIAGNOSTICO DE PROBLEMAS DE QUALIDADE (issue A01.2)
Estudo de caso: condicao cardiaca em criancas e adolescentes (UCMF_raw.csv)

Somente leitura: nada e corrigido aqui. Cada problema de qualidade que
justifica um tratamento no 02_limpeza.py e contado sobre a base BRUTA e
registrado com a sua origem:

  - "univariada"   -> o problema ja aparece no Modulo 1; a contagem e
                      conferida contra output/tabelas/resumo_univariado.csv
                      ou frequencias_categoricas.csv (coluna "confere");
  - "consistencia" -> regra que cruza colunas (duplicatas, idade x datas,
                      IMC x peso/altura, PAD x PAS...). Nao usa o alvo e nao
                      e analise bivariada: a relacao com o alvo e da A01.3;
  - "inspecao"     -> detalhamento de um problema da univariada (ex.: quais
                      datas nao parseaveis sao seriais do Excel).

As contagens sao independentes entre si (todas sobre as 12.873 linhas
brutas). As contagens na ordem em que os tratamentos sao aplicados ficam em
output/tabelas/limpeza_decisoes.csv, gerado pelo 02_limpeza.py.

Saida: output/tabelas/qualidade_diagnostico.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qualidade import (OUT_TAB, FAIXAS, IDADE_LIMITE_POPULACAO, ANO_LIMITE,
                       MAPA_SEXO, MAPA_ALVO, PPA_SEM_CLASSE, CATEG_GRAFIA,
                       carregar_bruto, colunas_conteudo, parse_data,
                       corrigir_seculo, idade_por_datas, fora_faixa, fc_intervalo,
                       imc_calculado, unificar_grafia, avaliar_antropometria,
                       descrever_limite_z, validar_tabelas_oms,
                       IDADE_SO_LIMITE_SUPERIOR)

os.makedirs(OUT_TAB, exist_ok=True)

df = carregar_bruto()
N = len(df)


def ler_univariada():
    """Tabelas do Modulo 1, para conferir as contagens (se existirem)."""
    p_res = os.path.join(OUT_TAB, "resumo_univariado.csv")
    p_frq = os.path.join(OUT_TAB, "frequencias_categoricas.csv")
    if not (os.path.exists(p_res) and os.path.exists(p_frq)):
        print("AVISO: tabelas da univariada ausentes; rode 01_analise_univariada.py "
              "para conferir as contagens.")
        return None, None
    res = pd.read_csv(p_res, sep=";", decimal=",", encoding="utf-8-sig")
    frq = pd.read_csv(p_frq, sep=";", decimal=",", encoding="utf-8-sig",
                      keep_default_na=False)
    return (res.set_index("variavel"),
            frq.set_index(["variavel", "categoria"])["frequencia"])


RESUMO, FREQ = ler_univariada()
achados = []


def registrar(variavel, problema, n, origem, ref=None, evidencia=""):
    """
    ref = ("resumo", variavel, coluna) ou ("freq", variavel, categoria):
    aponta o numero correspondente nas tabelas do Modulo 1.
    """
    valor_uni, fonte = None, ""
    if ref is not None:
        tipo, var, chave = ref
        if tipo == "resumo":
            fonte = f"resumo_univariado.csv [{var} · {chave}]"
            if RESUMO is not None:
                valor_uni = RESUMO.loc[var, chave]
        else:
            fonte = f"frequencias_categoricas.csv [{var} · {chave}]"
            if FREQ is not None:
                valor_uni = FREQ.get((var, chave), 0)
    if valor_uni is None:
        confere = "—"
    else:
        confere = "sim" if int(valor_uni) == int(n) else "NAO"
    achados.append({"variavel": variavel, "problema": problema, "n": int(n),
                    "origem": origem, "referencia_univariada": fonte,
                    "valor_univariada": "" if valor_uni is None else int(valor_uni),
                    "confere": confere, "evidencia": evidencia})


def lista(valores, k=8):
    v = pd.Series(valores).dropna().unique()[:k]
    return ", ".join(str(x) for x in v)


# =========================================================================
# 1. REGISTROS
# =========================================================================
registrar("ID", "identificador duplicado", df["ID"].duplicated().sum(), "inspecao",
          evidencia="mesma checagem impressa pelo 01_analise_univariada.py")
alvo_ausente = df["NORMAL X ANORMAL"].isna()
registrar("NORMAL X ANORMAL", "registro sem variável-alvo", alvo_ausente.sum(),
          "univariada", ("resumo", "NORMAL X ANORMAL", "ausentes"))

conteudo = colunas_conteudo(df)
dup = df.duplicated(conteudo)
grupos = df[df.duplicated(conteudo, keep=False)].groupby(conteudo, dropna=False).size()
maior = grupos.sort_values(ascending=False).index[0]
maior = dict(zip(conteudo, maior))
preenchidas = [c for c, v in maior.items() if pd.notna(v)]
registrar("(registro)", "linha idêntica a outra em todas as 20 colunas (difere só no ID)",
          dup.sum(), "consistencia",
          evidencia=(f"{len(grupos)} grupos; maior grupo com {grupos.max()} cópias, "
                     f"preenchido apenas em {', '.join(preenchidas)}"))
sem_alvo_dup = df[~alvo_ausente].duplicated(conteudo)
registrar("(registro)", "duplicatas restantes após remover registros sem alvo",
          sem_alvo_dup.sum(), "consistencia")


# =========================================================================
# 2. QUANTITATIVAS - sentinelas e valores fora da faixa de plausibilidade
# =========================================================================
num = {c: pd.to_numeric(df[c], errors="coerce") for c in FAIXAS}
SENTINELA_ZERO = ["Peso", "Altura", "IMC"]     # idade 0 e valida (recem-nascido)

for col, (lo, hi) in FAIXAS.items():
    x = num[col]
    registrar(col, "valor ausente", df[col].isna().sum(),
              "univariada", ("resumo", col, "ausentes"))
    texto = df[col].notna() & x.isna()
    if texto.any():
        registrar(col, "valor não numérico", texto.sum(), "univariada",
                  ("resumo", col, "nao_numericos"), lista(df.loc[texto, col]))
    if col in SENTINELA_ZERO:
        registrar(col, "zero usado como 'não medido'", (x == 0).sum(),
                  "univariada", ("resumo", col, "zeros"))
    if (x < 0).any():
        registrar(col, "valor negativo", (x < 0).sum(), "univariada",
                  ("resumo", col, "negativos"), lista(x[x < 0]))
    ff = fora_faixa(x, col)
    extremos = x[ff & (x != 0) & (x >= 0)]
    registrar(col, f"fora da faixa plausível [{lo}–{hi}] (inclui zeros e negativos)",
              ff.sum(), "univariada", ("resumo", col, "fora_faixa_clinica"),
              f"exemplos além de zero/negativos: {lista(extremos.sort_values())}"
              if len(extremos) else "")

peso, altura = num["Peso"], num["Altura"]
registrar("Peso/Altura", "peso e altura ambos zerados", ((peso == 0) & (altura == 0)).sum(),
          "consistencia")

fc_txt = df["FC"].notna() & num["FC"].isna()
fc_meio = fc_intervalo(df["FC"])
registrar("FC", "FC digitada como intervalo (ex.: '80-90')", (fc_txt & fc_meio.notna()).sum(),
          "inspecao", evidencia=lista(df.loc[fc_txt & fc_meio.notna(), "FC"], 16))

pas, pad = num["PA SISTOLICA"], num["PA DIASTOLICA"]
registrar("PA SISTOLICA/PA DIASTOLICA", "diastólica maior ou igual à sistólica",
          (pad >= pas).sum(), "consistencia",
          evidencia=lista((pas.astype(str) + "/" + pad.astype(str))[pad >= pas]))


# =========================================================================
# 3. DATAS
# =========================================================================
datas = {}
for col in ["Atendimento", "DN"]:
    d, serial = parse_data(df[col])
    datas[col] = d
    ddmm = pd.to_datetime(df[col], format="%d/%m/%y", errors="coerce")
    nao_parse = df[col].notna() & ddmm.isna()
    registrar(col, "data ausente", df[col].isna().sum(), "univariada",
              ("resumo", col, "ausentes"))
    registrar(col, "data não parseável como dd/mm/aa", nao_parse.sum(), "univariada",
              ("resumo", col, "nao_parseaveis"))
    registrar(col, "número serial do Excel no lugar da data", serial.sum(), "inspecao",
              evidencia=f"ex.: {lista(df.loc[serial, col], 3)} → "
              f"{lista(d[serial].dt.strftime('%d/%m/%Y'), 3)}; convertidas entre "
              f"{d[serial].min():%Y} e {d[serial].max():%Y}")
    malformada = df[col].notna() & d.isna()
    if malformada.any():
        registrar(col, "texto livre ou data malformada (irrecuperável)", malformada.sum(),
                  "inspecao", evidencia=lista(df.loc[malformada, col], 8))
    registrar(col, f"ano > {ANO_LIMITE} ("
              + ("século trocado pela leitura de 2 dígitos" if col == "DN"
                 else "fora do período de coleta") + ")",
              (ddmm.dt.year > ANO_LIMITE).sum(), "univariada",
              ("resumo", col, "datas_futuras"),
              lista(df.loc[ddmm.dt.year > ANO_LIMITE, col]))

at = datas["Atendimento"].mask(datas["Atendimento"].dt.year > ANO_LIMITE)
dn, seculo = corrigir_seculo(datas["DN"])
idade_seculo = idade_por_datas(at, dn)[seculo]
registrar("DN", "nascimento no futuro: século corrigido (−100 anos)", seculo.sum(),
          "inspecao",
          evidencia=f"anos {dn[seculo].dt.year.min()}–{dn[seculo].dt.year.max()} após a "
          f"correção; idade no atendimento {idade_seculo.min():.0f}–"
          f"{idade_seculo.max():.0f} anos (adultos)")
registrar("Atendimento/DN", "nascimento posterior ao atendimento", (dn > at).sum(),
          "consistencia")


# =========================================================================
# 4. COLUNAS DERIVADAS - IDADE, IMC e PPA contra as colunas de origem
# =========================================================================
idade = num["IDADE"]
calc = idade_por_datas(at, dn)
ambos = idade.notna() & calc.notna()
dif = (idade - calc)[ambos]
razao = (idade / calc)[ambos & (calc > 1)]
registrar("IDADE", "IDADE difere da calculada pelas datas em mais de 0,1 ano",
          (dif.abs() > 0.1).sum(), "consistencia",
          evidencia=f"de {ambos.sum()} comparáveis; IDADE/(Atendimento−DN) mediana = "
          f"{razao.median():.3f} (valor informado sistematicamente inflado)")
registrar("IDADE", "IDADE informada sem as duas datas (não recalculável)",
          (idade.notna() & calc.isna()).sum(), "consistencia")
registrar("IDADE", "idade calculada negativa (datas invertidas)", (calc < 0).sum(),
          "consistencia")
registrar("IDADE", f"idade calculada ≥ {IDADE_LIMITE_POPULACAO} anos "
          "(fora da população de 0 a 19 anos)",
          (calc >= IDADE_LIMITE_POPULACAO).sum(), "consistencia")
acima = idade > FAIXAS["IDADE"][1]
fica = acima & (calc < IDADE_LIMITE_POPULACAO)
registrar("IDADE", f"IDADE informada > {FAIXAS['IDADE'][1]} anos", acima.sum(),
          "consistencia",
          evidencia=f"recalculada pelas datas: {fica.sum()} caem em 0–19 anos e ficam; "
          f"{(acima & (calc >= IDADE_LIMITE_POPULACAO)).sum()} são adultos e saem")

imc = num["IMC"]
p_ok = peso.notna() & ~fora_faixa(peso, "Peso")
a_ok = altura.notna() & ~fora_faixa(altura, "Altura")
imc_c = imc_calculado(peso, altura)
comp = p_ok & a_ok & imc.notna()
registrar("IMC", "IMC difere de peso/altura² em mais de 1 kg/m²",
          ((imc - imc_c).abs()[comp] > 1).sum(), "consistencia",
          evidencia=f"de {comp.sum()} comparáveis; o IMC informado é arredondado para inteiro")
registrar("IMC", "IMC informado com peso ou altura inválidos (não verificável)",
          (imc.notna() & ~(p_ok & a_ok)).sum(), "consistencia")

# peso, altura e IMC contra as curvas da OMS (por idade e sexo)
erro_oms = validar_tabelas_oms()
av = avaliar_antropometria(peso.where(p_ok), altura.where(a_ok),
                           calc.where(calc >= 0), df["SEXO"].map(MAPA_SEXO))
ref_oms = (f"escore-z validado contra as tabelas da OMS (erro máx. {erro_oms:.3f}); "
           f"abaixo de {IDADE_SO_LIMITE_SUPERIOR:.0f} ano só o limite superior (prematuros)")
registrar("IDADE", "peso e altura coerentes entre si, mas implausíveis para a idade",
          av["idade_incompativel"].sum(), "consistencia",
          evidencia="ambos fora dos limites da OMS no mesmo sentido, com IMC plausível: "
          "o erro está nas datas")
for ind, col in [("peso", "Peso"), ("altura", "Altura"), ("imc", "IMC")]:
    registrar(col, f"implausível para idade e sexo ({descrever_limite_z(ind)})",
              av[ind].sum(), "consistencia", evidencia=ref_oms)

ppa = df["PPA"]
registrar("PPA", "erro de fórmula do Excel '#VALUE!'", (ppa == "#VALUE!").sum(),
          "univariada", ("freq", "PPA", "#VALUE!"))
registrar("PPA", "'#VALUE!' exatamente nos registros sem IDADE",
          ((ppa == "#VALUE!") & idade.isna()).sum(), "consistencia",
          evidencia=f"IDADE ausente em {idade.isna().sum()} registros")
registrar("PPA", "'Não Calculado' (sem classe de pressão)", (ppa == "Não Calculado").sum(),
          "univariada", ("freq", "PPA", "Não Calculado"))
registrar("PPA", "'Não Calculado' com PAS ou PAD ausente",
          ((ppa == "Não Calculado") & (pas.isna() | pad.isna())).sum(), "consistencia")


# =========================================================================
# 5. CATEGORICAS - ausentes, variantes de grafia e rotulos sem significado
# =========================================================================
# alvo fica de fora: seus ausentes ja estao em "registro sem variavel-alvo"
CATEG_AUSENTES = ["PULSOS", "PPA", "B2", "SOPRO", "HDA 1", "HDA2", "SEXO",
                  "MOTIVO1", "MOTIVO2", "Convenio"]
for col in CATEG_AUSENTES:
    registrar(col, "valor ausente", df[col].isna().sum(), "univariada",
              ("resumo", col, "ausentes"),
              f"{df[col].isna().mean() * 100:.1f}% dos registros".replace(".", ","))

for col in CATEG_GRAFIA:
    unif, _ = unificar_grafia(df[col])
    alterados = (df[col] != unif) & df[col].notna()
    if not alterados.any():
        continue
    registrar(col, "categorias distintas nos dados brutos", df[col].nunique(),
              "univariada", ("resumo", col, "categorias_distintas"))
    registrar(col, "registros com variante de grafia (caixa/acento/pontuação)",
              alterados.sum(), "consistencia",
              evidencia=f"{df[col].nunique()} → {unif.nunique()} categorias após unificar; "
              f"ex.: {lista(df.loc[alterados, col], 6)}")

sexo = df["SEXO"]
registrar("SEXO", "categorias distintas nos dados brutos", sexo.nunique(), "univariada",
          ("resumo", "SEXO", "categorias_distintas"), lista(sexo.unique()))
registrar("SEXO", "rótulo por extenso no lugar de M/F",
          sexo.isin(["Masculino", "masculino", "Feminino"]).sum(), "inspecao")
indet = sexo == "Indeterminado"
registrar("SEXO", "'Indeterminado' (sexo não informado)", indet.sum(), "univariada",
          ("freq", "SEXO", "Indeterminado"),
          f"{(indet & alvo_ausente).sum()} deles também sem alvo")
nao_mapeado = sexo.notna() & ~sexo.isin(MAPA_SEXO.keys())
registrar("SEXO", "rótulo fora do mapa conhecido", nao_mapeado.sum(), "inspecao")

alvo = df["NORMAL X ANORMAL"]
fora_padrao = alvo.notna() & ~alvo.isin(["Normal", "Anormal"])
registrar("NORMAL X ANORMAL", "rótulo fora do padrão Normal/Anormal", fora_padrao.sum(),
          "inspecao", evidencia=lista(alvo[fora_padrao]))


# =========================================================================
# SAIDA
# =========================================================================
def main():
    tab = pd.DataFrame(achados)
    tab.to_csv(os.path.join(OUT_TAB, "qualidade_diagnostico.csv"),
               index=False, encoding="utf-8-sig", sep=";")

    print(f"Registros brutos: {N}\n")
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_colwidth", 70):
        print(tab[["variavel", "problema", "n", "origem", "confere"]].to_string(index=False))

    divergentes = tab[tab["confere"] == "NAO"]
    conferidos = (tab["confere"] == "sim").sum()
    print(f"\n{conferidos} contagens conferidas com a univariada, "
          f"{len(divergentes)} divergentes.")
    if len(divergentes):
        print(divergentes.to_string(index=False))
        sys.exit(1)
    print(f"Tabela -> {os.path.join(OUT_TAB, 'qualidade_diagnostico.csv')}")


if __name__ == "__main__":
    main()
