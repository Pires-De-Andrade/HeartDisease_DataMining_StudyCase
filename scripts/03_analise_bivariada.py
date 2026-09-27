# -*- coding: utf-8 -*-
"""
MODULO 3 - ANALISE BIVARIADA
Estudo de caso: condicao cardiaca em criancas e adolescentes

Entrada: data/processed/UCMF_clean.csv (variaveis limpas, escala bruta, Modulo 2)
         unido por id a data/processed/UCMF_features.csv, do qual so se aproveitam
         as classes clinicas construidas no Modulo 4 (faixa_etaria, imc_classe,
         pa_classe, fc_classe, sopro_presente, n_queixas). Contrato em
         docs/contrato-dados.md.

Cada variavel independente e cruzada contra o alvo normal_x_anormal:
  - quantitativas -> teste de Mann-Whitney U (grupos nao normais, ja visto no
    Modulo 1) + correlacao rank-biserial como tamanho de efeito
  - categoricas    -> qui-quadrado de independencia + V de Cramer

Fora do escopo, pela mesma decisao registrada em docs/relatorio_a01.4.md:
id, convenio, hda2, ppa, atendimento, dn (irrelevantes, redundantes ou > 50%
ausentes). Categorias raras de variaveis de alta cardinalidade (hda1, motivo1,
motivo2) sao agregadas em "(outras)" antes do teste, para nao inflar o
qui-quadrado com contagens esperadas proximas de zero.

Saida: output/bivariate/*.png e output/tabelas/resumo_bivariado.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_theme import (apply_theme, strip_axes, figure_caption,
                       PRIMARY, CRITICAL, INK, INK_2, MUTED)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data", "processed", "UCMF_clean.csv")
FEATURES = os.path.join(BASE, "data", "processed", "UCMF_features.csv")
OUT_FIG = os.path.join(BASE, "output", "bivariate")
OUT_TAB = os.path.join(BASE, "output", "tabelas")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)

apply_theme()

ALVO = "normal_x_anormal"
GRUPOS = ["Normal", "Anormal"]
COR_GRUPO = {"Normal": PRIMARY, "Anormal": CRITICAL}

NOTA = ("Base limpa (Módulo 2), n = 11.401. Classes clínicas do Módulo 4. "
        "Ausentes excluídos par a par por variável.")


def br(n):
    """Formata inteiro no padrao pt-BR (separador de milhar '.')."""
    return f"{int(n):,}".replace(",", ".")


def num(x, casas=2):
    """Formata float no padrao pt-BR: milhar '.', decimal ','."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.{casas}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def slug(name):
    return (name.strip().lower().replace(" ", "_")
            .replace("á", "a").replace("é", "e").replace("í", "i")
            .replace("ó", "o").replace("ú", "u").replace("ç", "c"))


def forca_efeito(v):
    v = abs(v)
    if v < 0.10:
        return "desprezível"
    if v < 0.30:
        return "fraca"
    if v < 0.50:
        return "moderada"
    return "forte"


def agregar_raras(s, top):
    """Mantem as `top` categorias mais frequentes; o resto vira '(outras)'."""
    vc = s.value_counts()
    manter = set(vc.index[:top])
    return s.where(s.isna() | s.isin(manter), "(outras)")


# =========================================================================
# CARGA
# =========================================================================
clean = pd.read_csv(CLEAN)
feat = pd.read_csv(FEATURES)[["id", "faixa_etaria", "imc_classe", "pa_classe",
                              "fc_classe", "sopro_presente", "n_queixas"]]
df = clean.merge(feat, on="id", how="left")
N = len(df)

QUANT = {
    "peso":          ("Peso", "kg"),
    "altura":        ("Altura", "cm"),
    "imc":           ("IMC", "kg/m²"),
    "idade":         ("Idade", "anos"),
    "pa_sistolica":  ("PA sistólica", "mmHg"),
    "pa_diastolica": ("PA diastólica", "mmHg"),
    "fc":            ("FC", "bpm"),
}

CATEG_EXAME = {"sexo": None, "pulsos": None, "b2": None, "sopro": None}
CATEG_HISTORICO = {"hda1": 10, "motivo1": 10, "motivo2": 10}
CATEG_CLINICA = ["faixa_etaria", "imc_classe", "pa_classe", "fc_classe"]
CATEG_BINARIA = ["sopro_presente", "n_queixas"]


# =========================================================================
# TESTES
# =========================================================================
def teste_quant(col):
    d = df[[col, ALVO]].dropna()
    a = d.loc[d[ALVO] == "Normal", col]
    b = d.loc[d[ALVO] == "Anormal", col]
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - (2 * u) / (len(a) * len(b))          # correlacao rank-biserial
    return {"variavel": col, "tipo": "quantitativa", "n_valido": len(d),
            "teste": "Mann-Whitney U", "estatistica": u, "p_valor": p,
            "efeito": r, "mediana_normal": a.median(), "mediana_anormal": b.median()}


def teste_categ(col, top=None):
    s = df[col] if top is None else agregar_raras(df[col], top)
    d = pd.DataFrame({col: s, ALVO: df[ALVO]}).dropna()
    tab = pd.crosstab(d[col], d[ALVO])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    n = int(tab.values.sum())
    v = np.sqrt(chi2 / n)                         # k = min(2,c)-1 = 1 (alvo binario)
    return {"variavel": col, "tipo": "categórica", "n_valido": n,
            "teste": "Qui-quadrado", "estatistica": chi2, "p_valor": p,
            "efeito": v, "mediana_normal": np.nan, "mediana_anormal": np.nan}, tab


# =========================================================================
# FIGURAS
# =========================================================================
def fig_quantitativas():
    cols = list(QUANT)
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        titulo, unidade = QUANT[col]
        d = df[[col, ALVO]].dropna()
        dados = [d.loc[d[ALVO] == g, col] for g in GRUPOS]
        bp = ax.boxplot(dados, vert=True, widths=0.55, patch_artist=True,
                        tick_labels=GRUPOS,
                        flierprops=dict(marker="o", markersize=3, alpha=0.35,
                                        markerfacecolor=MUTED, markeredgecolor="none"),
                        medianprops=dict(color=INK, linewidth=1.8))
        for patch, g in zip(bp["boxes"], GRUPOS):
            patch.set_facecolor(COR_GRUPO[g])
            patch.set_alpha(0.35)
            patch.set_edgecolor(COR_GRUPO[g])
        ax.set_title(f"{titulo} ({unidade})", loc="left", fontsize=10.5, pad=8)
        strip_axes(ax, keep_y_grid=True, keep_x_grid=False)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Variáveis quantitativas × NORMAL X ANORMAL",
                 x=0.0, ha="left", fontsize=15, fontweight="bold", y=1.03)
    figure_caption(fig, NOTA + " Teste de Mann-Whitney U por variável "
                        "(estatística e p-valor na tabela-síntese).")
    fig.savefig(os.path.join(OUT_FIG, "quant_vs_alvo.png"), bbox_inches="tight")
    plt.close(fig)


def fig_categoricas(cols_dict, arquivo, titulo_fig, figsize):
    cols = list(cols_dict)
    fig, axes = plt.subplots(len(cols), 1, figsize=figsize)
    axes = np.atleast_1d(axes)
    taxa_global = (df[ALVO] == "Anormal").mean() * 100
    for i, (ax, col) in enumerate(zip(axes, cols)):
        top = cols_dict[col]
        s = df[col] if top is None else agregar_raras(df[col], top)
        d = pd.DataFrame({col: s, ALVO: df[ALVO]}).dropna()
        g = d.groupby(col)[ALVO].apply(lambda x: (x == "Anormal").mean() * 100)
        n = d.groupby(col).size()
        ordem = g.sort_values().index
        y = np.arange(len(ordem))
        ax.barh(y, g.loc[ordem], color=PRIMARY, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(ordem, fontsize=8.5)
        ax.axvline(taxa_global, color=INK, linestyle="--", linewidth=1)
        for yi, cat in zip(y, ordem):
            ax.text(g[cat] + 1.5, yi, f"{g[cat]:.0f}% (n={br(n[cat])})",
                    va="center", fontsize=8, color=INK_2)
        ax.set_xlim(0, 118)
        if i == len(cols) - 1:
            ax.set_xlabel("% Anormal")
        ax.set_title(col, loc="left", fontsize=10.5, pad=10)
        strip_axes(ax, keep_y_grid=False, keep_x_grid=True)
    fig.subplots_adjust(hspace=0.85)
    fig.suptitle(titulo_fig, x=0.0, ha="left", fontsize=15, fontweight="bold", y=1.01)
    figure_caption(fig, NOTA + " Linha tracejada = taxa global de Anormal "
                        f"({taxa_global:.0f}%).")
    fig.savefig(os.path.join(OUT_FIG, arquivo), bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# EXECUCAO
# =========================================================================
def main():
    print(f"Registros: {N} | Anormal: {(df[ALVO] == 'Anormal').mean():.1%}")

    resumo = []
    for col in QUANT:
        resumo.append(teste_quant(col))
    print("  [ok] testes quantitativos")

    for col, top in {**CATEG_EXAME, **CATEG_HISTORICO}.items():
        linha, _ = teste_categ(col, top=top)
        resumo.append(linha)
    for col in CATEG_CLINICA + CATEG_BINARIA:
        linha, _ = teste_categ(col)
        resumo.append(linha)
    print("  [ok] testes categóricos")

    tab = pd.DataFrame(resumo)
    tab["significativo (p<0,05)"] = np.where(tab["p_valor"] < 0.05, "sim", "não")
    tab["força do efeito"] = tab["efeito"].apply(forca_efeito)
    tab = tab.sort_values("efeito", key=abs, ascending=False)
    tab.to_csv(os.path.join(OUT_TAB, "resumo_bivariado.csv"), index=False,
               encoding="utf-8-sig", sep=";", decimal=",")
    print(tab[["variavel", "tipo", "n_valido", "p_valor", "efeito",
              "força do efeito"]].to_string(index=False))

    fig_quantitativas()
    print("  [ok] quant_vs_alvo.png")
    fig_categoricas(CATEG_EXAME, "cat_exame_vs_alvo.png",
                    "Achados de exame × NORMAL X ANORMAL", figsize=(9, 9.5))
    print("  [ok] cat_exame_vs_alvo.png")
    fig_categoricas(CATEG_HISTORICO, "cat_historico_vs_alvo.png",
                    "Histórico e motivo do encaminhamento × NORMAL X ANORMAL",
                    figsize=(9.5, 11))
    print("  [ok] cat_historico_vs_alvo.png")

    print(f"\nFiguras -> {OUT_FIG}")
    print(f"Tabelas -> {OUT_TAB}")
    print("\nClasses clínicas (faixa_etaria, imc_classe, pa_classe, fc_classe) "
          "já têm figura própria em output/transform/transform_classes_por_alvo.png "
          "(Módulo 4); aqui elas só entram na tabela-síntese com o teste formal.")


if __name__ == "__main__":
    main()
