# -*- coding: utf-8 -*-
"""
MODULO 2b - LIMPEZA (issue A01.2)
Estudo de caso: condicao cardiaca em criancas e adolescentes (UCMF_raw.csv)

Aplica os tratamentos justificados pelo 02_diagnostico_qualidade.py e entrega
a base no formato do contrato de dados (docs/contrato-dados.md):

  - corrige colunas existentes; nao cria nem remove colunas;
  - valores invalidos viram NaN; a limpeza NAO imputa (isso depende de
    estatistica da amostra e e feito na transformacao, ajustada so no treino);
  - colunas derivadas ja existentes (idade, imc) sao recalculadas pela formula.

Cada tratamento registra quantos registros afetou, NA ORDEM em que e
aplicado (por isso os numeros podem ser menores que os do diagnostico, que
conta tudo sobre a base bruta).

Saidas:
  data/processed/UCMF_clean.csv              base limpa (contrato)
  output/tabelas/limpeza_decisoes.csv        variavel; problema; tratamento; n_afetados
  output/tabelas/limpeza_mapa_grafias.csv    variante bruta -> grafia unificada
  output/limpeza/peso_altura_antes_depois.png  comparacao de duas variaveis
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qualidade import (BASE, OUT_TAB, FAIXAS, IDADE_LIMITE_POPULACAO, ANO_LIMITE,
                       SNAKE, MAPA_SEXO, MAPA_ALVO, PPA_SEM_CLASSE, CATEG_GRAFIA,
                       carregar_bruto, colunas_conteudo, parse_data,
                       corrigir_seculo, idade_por_datas, fora_faixa, fc_intervalo,
                       imc_calculado, unificar_grafia, chave_grafia,
                       avaliar_antropometria, descrever_limite_z)
from viz_theme import (apply_theme, strip_axes, figure_caption,
                       PRIMARY, CRITICAL, WARNING, INK, MUTED)

OUT_DATA = os.path.join(BASE, "data", "processed")
OUT_FIG = os.path.join(BASE, "output", "limpeza")
CSV_CLEAN = os.path.join(OUT_DATA, "UCMF_clean.csv")
for d in (OUT_DATA, OUT_TAB, OUT_FIG):
    os.makedirs(d, exist_ok=True)

apply_theme()

decisoes = []
mapas_grafia = []


def decidir(variavel, problema, tratamento, n):
    decisoes.append({"variavel": variavel, "problema": problema,
                     "tratamento": tratamento, "n_afetados": int(n)})


def br(n):
    return f"{int(n):,}".replace(",", ".")


def para_nan(df, col, mascara):
    df.loc[mascara, col] = np.nan
    return int(mascara.sum())


# =========================================================================
# 1. REGISTROS - sem alvo e duplicatas
# =========================================================================
def limpar_registros(df):
    sem_alvo = df["NORMAL X ANORMAL"].isna()
    decidir("NORMAL X ANORMAL", "registro sem variável-alvo",
            "registro removido (não serve para treino nem avaliação)", sem_alvo.sum())
    df = df[~sem_alvo]

    dup = df.duplicated(colunas_conteudo(df))
    decidir("(registro)", "linha idêntica a outra em todas as colunas exceto ID",
            "mantida a primeira ocorrência; cópias removidas", dup.sum())
    df = df[~dup]

    alvo = df["NORMAL X ANORMAL"]
    desconhecido = set(alvo.dropna()) - set(MAPA_ALVO)
    assert not desconhecido, f"rótulos de alvo não mapeados: {desconhecido}"
    fora = alvo != alvo.map(MAPA_ALVO)
    decidir("NORMAL X ANORMAL", "rótulo fora do padrão ('anormal', 'Normais')",
            "unificado em Normal/Anormal", fora.sum())
    df["NORMAL X ANORMAL"] = alvo.map(MAPA_ALVO)
    return df


# =========================================================================
# 2. DATAS E IDADE
# =========================================================================
def limpar_datas_idade(df):
    datas = {}
    for col in ["Atendimento", "DN"]:
        d, serial = parse_data(df[col])
        decidir(col, "número serial do Excel no lugar da data",
                "convertido para data (origem 30/12/1899)", serial.sum())
        malformada = df[col].notna() & d.isna()
        decidir(col, "texto livre ou data malformada",
                "NaN (data irrecuperável)", malformada.sum())
        if col == "DN":
            d, futuro = corrigir_seculo(d)
            decidir(col, f"ano > {ANO_LIMITE} (século trocado na leitura de 2 dígitos)",
                    "século corrigido (−100 anos)", futuro.sum())
        else:
            # recuar 100 anos poria a consulta antes do nascimento
            futuro = d.dt.year > ANO_LIMITE
            decidir(col, f"ano > {ANO_LIMITE} (fora do período de coleta)",
                    "NaN", futuro.sum())
            d = d.mask(futuro)
        datas[col] = d

    at, dn = datas["Atendimento"], datas["DN"]
    idade_bruta = pd.to_numeric(df["IDADE"], errors="coerce")
    idade = idade_por_datas(at, dn).round(2)
    invertida = idade < 0
    decidir("IDADE", "idade negativa (nascimento posterior ao atendimento)",
            "NaN na idade; datas e registro mantidos", invertida.sum())
    idade = idade.mask(invertida)
    decidir("IDADE", "IDADE informada sistematicamente acima da calculada pelas datas",
            "recalculada como (Atendimento − DN) / 365,25",
            (idade_bruta.notna() & idade.notna() & ((idade_bruta - idade).abs() > 0.01)).sum())
    decidir("IDADE", "IDADE informada sem as duas datas válidas",
            "NaN (fórmula de origem incorreta, valor não confiável)",
            (idade_bruta.notna() & idade.isna() & ~invertida).sum())

    df["Atendimento"] = at.dt.strftime("%Y-%m-%d")
    df["DN"] = dn.dt.strftime("%Y-%m-%d")
    df["IDADE"] = idade

    fora_pop = idade >= IDADE_LIMITE_POPULACAO
    decidir("IDADE", f"idade ≥ {IDADE_LIMITE_POPULACAO} anos (população do estudo: 0 a 19)",
            "registro removido", fora_pop.sum())
    return df[~fora_pop]


# =========================================================================
# 3. QUANTITATIVAS - sentinelas, faixas e colunas derivadas
# =========================================================================
def limpar_quantitativas(df):
    for col in ["Peso", "Altura"]:
        x = pd.to_numeric(df[col], errors="coerce")
        decidir(col, "zero usado como 'não medido'", "NaN", (x == 0).sum())
        decidir(col, f"negativo ou fora da faixa plausível {list(FAIXAS[col])}", "NaN",
                (fora_faixa(x, col) & (x != 0)).sum())
        df[col] = x.mask(fora_faixa(x, col))

    # segunda barreira: implausivel para a idade e o sexo (curvas da OMS)
    idade, sexo = df["IDADE"], df["SEXO"].map(MAPA_SEXO)
    av = avaliar_antropometria(df["Peso"], df["Altura"], idade, sexo)
    decidir("IDADE", "peso e altura coerentes entre si, mas implausíveis para a idade (OMS)",
            "NaN na idade (erro nas datas); peso e altura mantidos",
            av["idade_incompativel"].sum())
    df["IDADE"] = idade.mask(av["idade_incompativel"])
    for col in ["Peso", "Altura"]:
        decidir(col, f"implausível para idade e sexo ({descrever_limite_z(col.lower())}; "
                "abaixo de 1 ano só o limite superior, por causa de prematuros)",
                "NaN", av[col.lower()].sum())
        df[col] = df[col].mask(av[col.lower()])

    imc_bruto = pd.to_numeric(df["IMC"], errors="coerce")
    imc = imc_calculado(df["Peso"], df["Altura"]).round(2)
    decidir("IMC", "IMC informado arredondado para inteiro ou divergente",
            "recalculado como peso / altura² (m)",
            (imc_bruto.notna() & imc.notna() & ((imc_bruto - imc).abs() > 0.01)).sum())
    decidir("IMC", "IMC informado com peso ou altura inválidos",
            "NaN (não verificável)", (imc_bruto.notna() & imc.isna()).sum())
    decidir("IMC", f"IMC recalculado fora da faixa plausível {list(FAIXAS['IMC'])}", "NaN",
            fora_faixa(imc, "IMC").sum())
    imc = imc.mask(fora_faixa(imc, "IMC"))
    biv = av["imc"] & imc.notna()
    decidir("IMC", f"implausível para idade e sexo ({descrever_limite_z('imc')})",
            "NaN (peso e altura mantidos: não há como saber qual está errado)", biv.sum())
    df["IMC"] = imc.mask(biv)

    for col in ["PA SISTOLICA", "PA DIASTOLICA"]:
        x = pd.to_numeric(df[col], errors="coerce")
        decidir(col, f"fora da faixa plausível {list(FAIXAS[col])}", "NaN",
                fora_faixa(x, col).sum())
        df[col] = x.mask(fora_faixa(x, col))
    pas, pad = df["PA SISTOLICA"], df["PA DIASTOLICA"]
    invertida = pad >= pas
    decidir("PA SISTOLICA/PA DIASTOLICA", "diastólica maior ou igual à sistólica",
            "NaN nas duas", invertida.sum())
    df["PA SISTOLICA"] = pas.mask(invertida)
    df["PA DIASTOLICA"] = pad.mask(invertida)

    bruto = df["FC"]
    x = pd.to_numeric(bruto, errors="coerce")
    meio = fc_intervalo(bruto)
    intervalo = x.isna() & meio.notna()
    decidir("FC", "FC digitada como intervalo ('80-90', '156 a 168')",
            "substituída pelo ponto médio", intervalo.sum())
    x = x.fillna(meio)
    decidir("FC", "texto não numérico restante", "NaN", (bruto.notna() & x.isna()).sum())
    decidir("FC", f"fora da faixa plausível {list(FAIXAS['FC'])}", "NaN",
            fora_faixa(x, "FC").sum())
    df["FC"] = x.mask(fora_faixa(x, "FC"))
    return df


# =========================================================================
# 4. CATEGORICAS
# =========================================================================
def limpar_categoricas(df):
    for rotulo in PPA_SEM_CLASSE:
        m = df["PPA"] == rotulo
        motivo = ("erro de fórmula do Excel (ocorre onde falta IDADE)"
                  if rotulo == "#VALUE!" else "classe não calculada")
        decidir("PPA", f"'{rotulo}': {motivo}", "NaN (não é uma classe de pressão)",
                para_nan(df, "PPA", m))

    sexo = df["SEXO"]
    desconhecido = set(sexo.dropna()) - set(MAPA_SEXO)
    assert not desconhecido, f"rótulos de sexo não mapeados: {desconhecido}"
    decidir("SEXO", "rótulo por extenso ('Masculino', 'masculino', 'Feminino')",
            "unificado em M/F", sexo.isin(["Masculino", "masculino", "Feminino"]).sum())
    decidir("SEXO", "'Indeterminado' (sexo não informado)", "NaN",
            (sexo == "Indeterminado").sum())
    df["SEXO"] = sexo.map(MAPA_SEXO)

    for col in CATEG_GRAFIA:
        unif, mapa = unificar_grafia(df[col])
        alterados = df[col].notna() & (df[col] != unif)
        if alterados.any():
            decidir(col, "variantes de grafia (caixa, acento, pontuação)",
                    f"unificadas na grafia mais frequente "
                    f"({df[col].nunique()} → {unif.nunique()} categorias)", alterados.sum())
            mudou = mapa[mapa["valor"] != mapa["canonica"]]
            mapas_grafia.append(mudou.assign(variavel=col))
        df[col] = unif
    return df


# =========================================================================
# 5. CONTRATO - tipos, nomes e verificacao
# =========================================================================
NUMERICAS = ["peso", "altura", "imc", "idade", "pa_sistolica", "pa_diastolica", "fc"]


def aplicar_contrato(df):
    df = df.rename(columns=SNAKE)[list(SNAKE.values())]
    df["id"] = df["id"].astype(int)
    for c in NUMERICAS:
        df[c] = df[c].astype(float)
    return df.reset_index(drop=True)


def verificar_contrato(df):
    """Falha se a base limpa violar docs/contrato-dados.md."""
    assert list(df.columns) == list(SNAKE.values()), "colunas diferentes das 21 originais"
    assert df["id"].is_unique
    for c in NUMERICAS:
        assert df[c].dtype == float, f"{c} não é float"
        assert not (df[c] <= 0).any() or c == "idade", f"{c} com zero/negativo"
        orig = {v: k for k, v in SNAKE.items()}[c]
        if c != "idade":       # idade: anos completos, verificada abaixo
            assert not fora_faixa(df[c], orig).any(), f"{c} fora da faixa"
    assert (df["idade"].dropna() >= 0).all()
    assert (df["idade"].dropna() < IDADE_LIMITE_POPULACAO).all()
    av = avaliar_antropometria(df["peso"], df["altura"], df["idade"], df["sexo"])
    for k in ["idade_incompativel", "peso", "altura"]:
        assert not av[k].any(), f"{k} implausível pela OMS na base limpa"
    # IMC implausivel foi anulado mantendo peso e altura: checa o IMC gravado
    assert not (av["imc"] & df["imc"].notna()).any(), "imc implausível pela OMS"
    for c in ["atendimento", "dn"]:
        ok = df[c].isna() | df[c].str.fullmatch(r"\d{4}-\d{2}-\d{2}")
        assert ok.all(), f"{c} fora do formato ISO"
    assert df["normal_x_anormal"].isin(["Normal", "Anormal"]).all()
    assert df["sexo"].dropna().isin(["M", "F"]).all()
    texto = df.select_dtypes(exclude="number")
    for sentinela in ["#VALUE!", "Não Calculado", "Indeterminado"]:
        assert not (texto == sentinela).any().any(), f"sentinela '{sentinela}' presente"
    assert not df.duplicated([c for c in df.columns if c != "id"]).any()
    for c in texto.columns.difference(["atendimento", "dn"]):
        v = df[c].dropna().unique()
        assert len(v) == len({chave_grafia(x) for x in v}), f"{c} com grafias duplicadas"


# =========================================================================
# 6. COMPARACAO DE DUAS VARIAVEIS - peso x altura antes e depois
# =========================================================================
def plot_peso_altura(bruto, limpo):
    p = pd.to_numeric(bruto["Peso"], errors="coerce")
    a = pd.to_numeric(bruto["Altura"], errors="coerce")
    tem_par = p.notna() & a.notna()
    at = parse_data(bruto["Atendimento"])[0]
    dn = corrigir_seculo(parse_data(bruto["DN"])[0])[0]
    idade = idade_por_datas(at, dn)
    ff_p, ff_a = fora_faixa(p, "Peso"), fora_faixa(a, "Altura")
    av = avaliar_antropometria(p.mask(ff_p), a.mask(ff_a), idade.where(idade >= 0),
                               bruto["SEXO"].map(MAPA_SEXO))
    invalido = tem_par & (ff_p | ff_a | av["peso"] | av["altura"])
    valido = tem_par & ~invalido
    zerados = int(((p == 0) & (a == 0)).sum())

    lp = limpo[["peso", "altura", "imc"]].dropna(subset=["peso", "altura"])
    sem_imc = lp["imc"].isna()        # par incoerente: so o IMC foi anulado

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    axA.scatter(a[valido], p[valido], s=6, color=PRIMARY, alpha=0.35,
                linewidths=0, label=f"par plausível ({br(valido.sum())})")
    axA.scatter(a[invalido], p[invalido], s=18, color=CRITICAL, marker="x",
                linewidths=0.9, label=f"par com valor inválido ou implausível\npara a idade ({br(invalido.sum())})")
    axA.annotate(f"{br(zerados)} registros com\npeso = 0 e altura = 0",
                 xy=(0, 0), xytext=(22, 95), textcoords="data", fontsize=9,
                 color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    axA.set_title(f"A · Dados brutos ({br(tem_par.sum())} pares)", loc="left", pad=10)
    axA.legend(loc="upper right")

    axB.scatter(lp.loc[~sem_imc, "altura"], lp.loc[~sem_imc, "peso"], s=6, color=PRIMARY,
                alpha=0.35, linewidths=0, label=f"par coerente ({br((~sem_imc).sum())})")
    axB.scatter(lp.loc[sem_imc, "altura"], lp.loc[sem_imc, "peso"], s=22, marker="^",
                facecolors="none", edgecolors=WARNING, linewidths=1.1,
                label=f"par incoerente mantido, IMC anulado\n"
                      f"(não se sabe qual medida errou) ({br(sem_imc.sum())})")
    axB.set_title(f"B · Após a limpeza ({br(len(lp))} pares)", loc="left", pad=10)
    axB.legend(loc="upper right")

    for ax in (axA, axB):
        ax.set_ylim(-50, 205)             # folga no topo para a legenda
        ax.set_xlabel("Altura (cm)")
        strip_axes(ax, keep_y_grid=True, keep_x_grid=True)
    axA.set_ylabel("Peso (kg)")

    fig.suptitle("Peso × altura — antes e depois da limpeza",
                 x=0.0, ha="left", fontsize=15, fontweight="bold", y=1.04)
    fig.text(0.0, 0.975,
             "Mesmos eixos nos dois painéis. Zeros, valores fora das faixas plausíveis e "
             "valores implausíveis para a idade (curvas da OMS) viram ausentes; "
             "a nuvem restante segue a curva de crescimento.",
             transform=fig.transFigure, fontsize=9.5, color=MUTED, va="top")
    figure_caption(fig, "Fonte: UCMF_raw.csv (A) e data/processed/UCMF_clean.csv (B). "
                        "Pares com peso ou altura ausente não aparecem. B tem menos pares\n"
                        "que os plausíveis de A porque registros sem alvo, duplicados e "
                        "fora da faixa etária foram removidos por inteiro.")
    out = os.path.join(OUT_FIG, "peso_altura_antes_depois.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# =========================================================================
# EXECUCAO
# =========================================================================
def main():
    bruto = carregar_bruto()
    n0 = len(bruto)

    df = limpar_registros(bruto.copy())
    df = limpar_datas_idade(df)
    df = limpar_quantitativas(df)
    df = limpar_categoricas(df)

    dup = df.duplicated(colunas_conteudo(df))
    decidir("(registro)", "linhas que só diferiam na grafia (ex.: 'sistólico' × 'Sistólico')",
            "mantida a primeira ocorrência; cópias removidas", dup.sum())
    df = df[~dup]

    df = aplicar_contrato(df)
    verificar_contrato(df)

    original = {v: k for k, v in SNAKE.items()}
    for c in df.columns:
        decidir(original[c], "valor ausente na base limpa (bruto + anulado acima)",
                "mantido como NaN; imputação fica na transformação, ajustada só no treino",
                df[c].isna().sum())

    df.to_csv(CSV_CLEAN, index=False, encoding="utf-8")
    tab = pd.DataFrame(decisoes)
    tab.to_csv(os.path.join(OUT_TAB, "limpeza_decisoes.csv"),
               index=False, encoding="utf-8-sig", sep=";")
    if mapas_grafia:
        (pd.concat(mapas_grafia)[["variavel", "valor", "canonica", "n"]]
         .rename(columns={"valor": "variante_bruta", "canonica": "grafia_unificada",
                          "n": "n_registros"})
         .to_csv(os.path.join(OUT_TAB, "limpeza_mapa_grafias.csv"),
                 index=False, encoding="utf-8-sig", sep=";"))
    fig = plot_peso_altura(bruto, df)

    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_colwidth", 80):
        print(tab.to_string(index=False))
    print(f"\nRegistros: {n0} brutos -> {len(df)} limpos "
          f"({n0 - len(df)} removidos). Contrato verificado.")
    print(f"Base   -> {CSV_CLEAN}")
    print(f"Figura -> {fig}")


if __name__ == "__main__":
    main()
