# -*- coding: utf-8 -*-
"""
MODULO 1 - ANALISE UNIVARIADA
Estudo de caso: condicao cardiaca em criancas e adolescentes (UCMF_raw.csv)

PREMISSA METODOLOGICA
---------------------
Todos os graficos deste modulo sao gerados sobre os dados BRUTOS, sem qualquer
limpeza, padronizacao de caixa ou remocao de outliers. O objetivo da analise
univariada dentro do processo KDD e justamente EXPOR os problemas de qualidade
que o Modulo 2 (identificacao de problemas) ira catalogar. Por isso:

  - variantes de caixa NAO sao unificadas ("Sistolico" vs "sistolico");
  - valores sentinela (0, negativos) NAO sao removidos;
  - valores ausentes aparecem como uma categoria propria "(ausente)",
    nunca sao descartados silenciosamente.

As "faixas clinicas plausiveis" usadas no 3o painel das variaveis quantitativas
sao apenas uma JANELA DE VISUALIZACAO para recuperar a forma real da
distribuicao, que os outliers achatam no 1o painel. Nao constituem limpeza.

Saida: output/univariate/*.png e output/tabelas/*.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_theme import (apply_theme, strip_axes, figure_caption,
                       PRIMARY, CRITICAL, WARNING, ABSENT,
                       INK, INK_2, MUTED, SURFACE, SERIES)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "UCMF_raw.csv")
OUT_FIG = os.path.join(BASE, "output", "univariate")
OUT_TAB = os.path.join(BASE, "output", "tabelas")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)

apply_theme()

NOTA = ("Dados brutos, sem tratamento. Ausentes exibidos explicitamente. "
        "Fonte: UCMF_raw.csv (n = 12.873 registros).")


def br(n):
    """Formata inteiro no padrao pt-BR (separador de milhar '.')."""
    return f"{int(n):,}".replace(",", ".")


def pct(x):
    return f"{x:.1f}%".replace(".", ",")


def num(x, casas=2):
    """Formata float no padrao pt-BR: milhar '.', decimal ','."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.{casas}f}"                      # 1,234.57
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def slug(name):
    return (name.strip().lower().replace(" ", "_").replace("x_", "x_")
            .replace("á", "a").replace("é", "e").replace("í", "i")
            .replace("ó", "o").replace("ú", "u").replace("ç", "c"))


# =========================================================================
# CARGA - tudo como texto, para nao perder nenhuma anomalia na leitura
# =========================================================================
df = pd.read_csv(CSV, dtype=str, keep_default_na=False, na_values=[""])
N = len(df)

QUANT = {
    #  variavel          : (faixa clinica plausivel, unidade)
    "Peso":           ((0.5, 150), "kg"),
    "Altura":         ((30, 200), "cm"),
    "IMC":            ((8, 45), "kg/m²"),
    "IDADE":          ((0, 19), "anos"),
    "PA SISTOLICA":   ((50, 200), "mmHg"),
    "PA DIASTOLICA":  ((20, 150), "mmHg"),
    "FC":             ((40, 220), "bpm"),
}

CATEG = ["PULSOS", "PPA", "NORMAL X ANORMAL", "B2", "SOPRO",
         "HDA 1", "HDA2", "SEXO", "MOTIVO1", "MOTIVO2"]

ORDEM_PPA = ["Normal", "Pre-Hipertensão PAS", "Pre-Hipertensão PAD",
             "HAS-1 PAS", "HAS-1 PAD", "HAS-2 PAS", "HAS-2 PAD",
             "Não Calculado", "#VALUE!"]

TEMPORAL = ["Atendimento", "DN"]


# =========================================================================
# 1. VARIAVEIS QUANTITATIVAS - figura de 3 paineis
# =========================================================================
def plot_quantitativa(col, faixa, unidade):
    bruto = df[col]
    serie = pd.to_numeric(bruto, errors="coerce")

    n_ausente = bruto.isna().sum()
    n_texto = int((bruto.notna() & serie.isna()).sum())   # strings nao numericas
    v = serie.dropna()
    lo, hi = faixa
    dentro = v[(v >= lo) & (v <= hi)]
    n_zero = int((v == 0).sum())
    n_neg = int((v < 0).sum())
    n_fora = int(len(v) - len(dentro))

    fig = plt.figure(figsize=(11, 9.2))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 3], hspace=0.55, figure=fig)

    # --- Painel A: histograma na amplitude bruta completa -----------------
    axA = fig.add_subplot(gs[0])
    if len(v):
        counts, bins, patches = axA.hist(v, bins=60, color=PRIMARY,
                                         edgecolor=SURFACE, linewidth=0.6)
        # marca em vermelho as barras cujo conteudo esta fora da faixa clinica
        for p, b0, b1 in zip(patches, bins[:-1], bins[1:]):
            centro = (b0 + b1) / 2
            if centro < lo or centro > hi:
                p.set_facecolor(CRITICAL)
    axA.set_title(f"A · {col} — amplitude bruta completa", loc="left", pad=26)
    axA.set_xlabel(f"{col} ({unidade})")
    axA.set_ylabel("Frequência")
    axA.set_yscale("log")
    strip_axes(axA)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PRIMARY),
               plt.Rectangle((0, 0), 1, 1, fc=CRITICAL)]
    axA.legend(handles, [f"dentro da faixa plausível [{lo}–{hi}]",
                         "fora da faixa plausível (suspeito/inválido)"],
               loc="upper right")
    axA.text(0.0, 1.02, "Escala vertical logarítmica: sem ela, caudas de poucos "
             "registros ficariam invisíveis.",
             transform=axA.transAxes, fontsize=8.5, color=MUTED, va="bottom")

    # --- Painel B: boxplot ------------------------------------------------
    axB = fig.add_subplot(gs[1])
    if len(v):
        bp = axB.boxplot(v, vert=False, widths=0.45, patch_artist=True,
                         flierprops=dict(marker="o", markersize=3.5,
                                         markerfacecolor=CRITICAL,
                                         markeredgecolor="none", alpha=0.45),
                         medianprops=dict(color=INK, linewidth=1.8),
                         boxprops=dict(facecolor=PRIMARY, alpha=0.30,
                                       edgecolor=PRIMARY, linewidth=1.4),
                         whiskerprops=dict(color=PRIMARY, linewidth=1.4),
                         capprops=dict(color=PRIMARY, linewidth=1.4))
    axB.set_title(f"B · {col} — dispersão e outliers (critério 1,5 × IIQ)",
                  loc="left", pad=8)
    axB.set_xlabel(f"{col} ({unidade})")
    axB.set_yticks([])
    strip_axes(axB, keep_y_grid=False, keep_x_grid=True)

    # --- Painel C: histograma na janela clinica plausivel -----------------
    axC = fig.add_subplot(gs[2])
    if len(dentro):
        axC.hist(dentro, bins=45, color=PRIMARY,
                 edgecolor=SURFACE, linewidth=0.6)
        med = dentro.median()
        axC.axvline(med, color=INK, linewidth=1.6, linestyle="--")
        axC.annotate(f"mediana na janela = {num(med, 1)}",
                     xy=(med, axC.get_ylim()[1] * 0.92),
                     xytext=(8, 0), textcoords="offset points",
                     fontsize=9, color=INK, fontweight="semibold")
    axC.set_title(f"C · {col} — janela clínica plausível [{lo}–{hi}] "
                  f"({br(len(dentro))} registros)", loc="left", pad=26)
    axC.set_xlabel(f"{col} ({unidade})")
    axC.set_ylabel("Frequência")
    strip_axes(axC)
    axC.text(0.0, 1.02, "Mesma variável, recorte de visualização — revela a forma "
             "real da distribuição que o painel A achata. Não é limpeza.",
             transform=axC.transAxes, fontsize=8.5, color=MUTED, va="bottom")

    # --- caixa de diagnostico --------------------------------------------
    linhas = [
        "VOLUME",
        f"  n total        {br(N)}",
        f"  ausentes       {br(n_ausente)}  ({pct(n_ausente / N * 100)})",
        f"  não numéricos  {br(n_texto)}",
        f"  válidos lidos  {br(len(v))}",
        "",
        "POSIÇÃO E DISPERSÃO",
        f"  mínimo         {num(v.min()) if len(v) else '—'}",
        f"  máximo         {num(v.max()) if len(v) else '—'}",
        f"  média          {num(v.mean()) if len(v) else '—'}",
        f"  mediana        {num(v.median()) if len(v) else '—'}",
        f"  desvio-padrão  {num(v.std()) if len(v) else '—'}",
        "",
        "SINAIS DE PROBLEMA",
        f"  zeros          {br(n_zero)}",
        f"  negativos      {br(n_neg)}",
        f"  fora da faixa  {br(n_fora)}  ({pct(n_fora / max(len(v), 1) * 100)})",
    ]
    fig.text(1.005, 0.97, "\n".join(linhas),
             transform=fig.transFigure, fontsize=8.6, color=INK_2,
             va="top", ha="left", family="monospace", linespacing=1.6)

    fig.suptitle(f"{col} — distribuição univariada (dados brutos)",
                 x=0.0, ha="left", fontsize=15, fontweight="bold",
                 color=INK, y=1.005)
    figure_caption(fig, NOTA)
    out = os.path.join(OUT_FIG, f"quant_{slug(col)}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {"variavel": col, "tipo": "quantitativa", "n_total": N,
            "ausentes": n_ausente, "pct_ausentes": round(n_ausente / N * 100, 2),
            "nao_numericos": n_texto, "validos": len(v),
            "min": v.min() if len(v) else np.nan,
            "max": v.max() if len(v) else np.nan,
            "media": round(v.mean(), 3) if len(v) else np.nan,
            "mediana": v.median() if len(v) else np.nan,
            "desvio": round(v.std(), 3) if len(v) else np.nan,
            "zeros": n_zero, "negativos": n_neg,
            "fora_faixa_clinica": n_fora}


# =========================================================================
# 2. VARIAVEIS CATEGORICAS - barras horizontais com ausentes explicitos
# =========================================================================
def plot_categorica(col, ordem=None, top=None):
    s = df[col]
    vc = s.value_counts(dropna=True)
    if ordem:
        vc = vc.reindex([c for c in ordem if c in vc.index]).dropna()
    total_cat = len(vc)
    resto = 0
    if top and len(vc) > top:
        resto = int(vc.iloc[top:].sum())
        vc = vc.iloc[:top]

    n_aus = int(s.isna().sum())
    rotulos = list(vc.index)
    valores = list(vc.values)
    cores = [PRIMARY] * len(valores)
    if resto:
        rotulos.append(f"(outras {total_cat - top} categorias)")
        valores.append(resto)
        cores.append(SERIES[1])
    rotulos.append("(ausente)")
    valores.append(n_aus)
    cores.append(ABSENT)

    altura = max(3.2, 0.42 * len(rotulos) + 2.2)
    fig, ax = plt.subplots(figsize=(10.5, altura))
    y = np.arange(len(rotulos))[::-1]
    ax.barh(y, valores, color=cores, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(rotulos, fontsize=9.5, color=INK_2)
    ax.set_xlabel("Número de registros")
    lim = max(valores) if valores else 1
    ax.set_xlim(0, lim * 1.18)
    for yi, val in zip(y, valores):
        ax.text(val + lim * 0.012, yi, f"{br(val)}  ({pct(val / N * 100)})",
                va="center", fontsize=9, color=INK_2)
    strip_axes(ax, keep_y_grid=False, keep_x_grid=True)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=PRIMARY),
               plt.Rectangle((0, 0), 1, 1, fc=ABSENT)]
    labels = ["categoria observada", "valor ausente (NaN)"]
    if resto:
        handles.insert(1, plt.Rectangle((0, 0), 1, 1, fc=SERIES[1]))
        labels.insert(1, "cauda agregada")
    ax.legend(handles, labels, loc="lower right")

    sub = f"{total_cat} categorias distintas nos dados brutos"
    ax.set_title(f"{col} — distribuição de frequências (dados brutos)\n",
                 loc="left", fontsize=14, fontweight="bold", pad=22)
    ax.text(0.0, 1.015, sub, transform=ax.transAxes,
            fontsize=9.5, color=MUTED, va="bottom")
    figure_caption(fig, NOTA)
    fig.savefig(os.path.join(OUT_FIG, f"cat_{slug(col)}.png"), bbox_inches="tight")
    plt.close(fig)
    return {"variavel": col, "tipo": "qualitativa", "n_total": N,
            "categorias_distintas": total_cat, "ausentes": n_aus,
            "pct_ausentes": round(n_aus / N * 100, 2),
            "categoria_moda": vc.index[0] if len(vc) else None,
            "freq_moda": int(vc.iloc[0]) if len(vc) else 0}


# =========================================================================
# 3. CONVENIO - alta cardinalidade: top-20 + curva de cobertura acumulada
# =========================================================================
def plot_convenio():
    col = "Convenio"
    s = df[col]
    vc = s.value_counts()
    n_aus = int(s.isna().sum())

    fig = plt.figure(figsize=(12.5, 8.6))
    gs = GridSpec(1, 2, width_ratios=[1.25, 1], wspace=0.42, figure=fig)

    # -- esquerda: top 20
    ax1 = fig.add_subplot(gs[0])
    top = vc.head(20)
    y = np.arange(len(top))[::-1]
    ax1.barh(y, top.values, color=PRIMARY, height=0.68)
    ax1.set_yticks(y)
    ax1.set_yticklabels(top.index, fontsize=9)
    ax1.set_xlim(0, top.max() * 1.22)
    for yi, val in zip(y, top.values):
        ax1.text(val + top.max() * 0.015, yi, br(val), va="center",
                 fontsize=8.5, color=INK_2)
    ax1.set_xlabel("Número de registros")
    ax1.set_title("A · 20 convênios mais frequentes", loc="left", pad=10)
    strip_axes(ax1, keep_y_grid=False, keep_x_grid=True)

    # -- direita: cobertura acumulada (grafico proprio, escala propria)
    ax2 = fig.add_subplot(gs[1])
    cum = vc.cumsum() / vc.sum() * 100
    x = np.arange(1, len(cum) + 1)
    ax2.plot(x, cum.values, color=PRIMARY, linewidth=2)
    ax2.fill_between(x, cum.values, color=PRIMARY, alpha=0.10)
    for marca in (10, 20, 50):
        if marca <= len(cum):
            v = cum.values[marca - 1]
            ax2.plot([marca], [v], "o", markersize=8, color=PRIMARY,
                     markeredgecolor=SURFACE, markeredgewidth=2)
            ax2.annotate(f"{marca} categorias → {v:.0f}%".replace(".", ","),
                         xy=(marca, v), xytext=(12, -14),
                         textcoords="offset points", fontsize=9, color=INK_2)
    ax2.set_xlabel("Número de categorias (ordenadas por frequência)")
    ax2.set_ylabel("Cobertura acumulada dos registros (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("B · Concentração da cauda longa", loc="left", pad=10)
    strip_axes(ax2)

    singletons = int((vc == 1).sum())
    nota = (f"{len(vc)} categorias distintas · {br(n_aus)} ausentes "
            f"({pct(n_aus / N * 100)}) · {singletons} categorias com "
            f"um único registro. A cauda inclui variantes de grafia do mesmo "
            f"convênio (ex.: 'Camed'/'CAMED', 'Medial'/'MEDIAL').")
    fig.suptitle("Convenio — variável nominal de alta cardinalidade (dados brutos)",
                 x=0.0, ha="left", fontsize=15, fontweight="bold", y=1.03)
    fig.text(0.0, 0.985, nota, transform=fig.transFigure, fontsize=9.5,
             color=MUTED, va="top", ha="left")
    figure_caption(fig, NOTA)
    fig.savefig(os.path.join(OUT_FIG, "cat_convenio.png"), bbox_inches="tight")
    plt.close(fig)
    return {"variavel": col, "tipo": "qualitativa (alta cardinalidade)",
            "n_total": N, "categorias_distintas": len(vc), "ausentes": n_aus,
            "pct_ausentes": round(n_aus / N * 100, 2),
            "categoria_moda": vc.index[0], "freq_moda": int(vc.iloc[0])}


# =========================================================================
# 4. VARIAVEIS TEMPORAIS
# =========================================================================
def plot_temporais():
    ANO_LIMITE = 2010          # ultimo ano plausivel de coleta neste conjunto
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    resumo = []
    for ax, col, titulo in zip(
            axes, TEMPORAL,
            ["A · Atendimento — ano da visita clínica",
             "B · DN — ano de nascimento do paciente"]):
        d = pd.to_datetime(df[col], format="%d/%m/%y", errors="coerce")
        n_aus = int(df[col].isna().sum())
        n_inval = int((df[col].notna() & d.isna()).sum())
        anos = d.dt.year.dropna().astype(int)
        n_futuro = int((anos > ANO_LIMITE).sum())
        vc = anos.value_counts().sort_index()
        cores = [CRITICAL if a > ANO_LIMITE else PRIMARY for a in vc.index]
        ax.bar(vc.index, vc.values, color=cores, width=0.8,
               edgecolor=SURFACE, linewidth=0.5)
        ax.set_yscale("log")
        ax.set_title(titulo, loc="left", pad=26)
        ax.set_xlabel("Ano")
        ax.set_ylabel("Número de registros (escala log)")
        strip_axes(ax)
        ax.text(0.0, 1.02,
                f"ausentes: {br(n_aus)} ({pct(n_aus / N * 100)})  ·  "
                f"não parseáveis: {br(n_inval)}  ·  "
                f"intervalo lido: {anos.min()}–{anos.max()}",
                transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
        if n_futuro:
            ax.annotate(f"{br(n_futuro)} registros com ano > {ANO_LIMITE}\n"
                        f"(datas impossíveis)",
                        xy=(vc.index.max(), vc.values[-1]),
                        xytext=(-10, 55), textcoords="offset points",
                        ha="right", fontsize=9, color=CRITICAL,
                        fontweight="semibold",
                        arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.2))
        handles = [plt.Rectangle((0, 0), 1, 1, fc=PRIMARY),
                   plt.Rectangle((0, 0), 1, 1, fc=CRITICAL)]
        ax.legend(handles, [f"ano ≤ {ANO_LIMITE} (plausível)",
                            f"ano > {ANO_LIMITE} (impossível)"],
                  loc="upper left", fontsize=8.5)
        resumo.append({"variavel": col, "tipo": "temporal", "n_total": N,
                       "ausentes": n_aus, "pct_ausentes": round(n_aus / N * 100, 2),
                       "nao_parseaveis": n_inval, "datas_futuras": n_futuro,
                       "ano_min": int(anos.min()), "ano_max": int(anos.max())})
    fig.suptitle("Variáveis temporais — cobertura do período de coleta (dados brutos)",
                 x=0.0, ha="left", fontsize=15, fontweight="bold", y=1.16)
    fig.text(0.0, 1.10,
             "O arquivo grava o ano com 2 dígitos (dd/mm/aa): o século é ambíguo e a "
             "conversão automática joga anos como '66' para 2066. A janela de\n"
             "conversão correta é uma decisão do Módulo 2. Escala logarítmica para "
             "tornar visíveis os anos com pouquíssimos registros.",
             transform=fig.transFigure, fontsize=9.5, color=MUTED, va="top")
    plt.tight_layout()
    figure_caption(fig, NOTA)
    fig.savefig(os.path.join(OUT_FIG, "temporal_datas.png"), bbox_inches="tight")
    plt.close(fig)
    return resumo


# =========================================================================
# 5. PANORAMA DE AUSENTES (visao cruzada de todas as variaveis)
# =========================================================================
def plot_ausentes():
    aus = (df.isna().sum() / N * 100).sort_values(ascending=True)
    cores, faixa = [], []
    for v in aus.values:
        if v > 50:
            cores.append(CRITICAL); faixa.append("alta")
        elif v >= 10:
            cores.append(WARNING); faixa.append("moderada")
        else:
            cores.append(PRIMARY); faixa.append("baixa")

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    y = np.arange(len(aus))
    ax.barh(y, aus.values, color=cores, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(aus.index, fontsize=10, color=INK_2)
    ax.set_xlabel("Registros ausentes (% do total)")
    ax.set_xlim(0, 108)
    for yi, v in zip(y, aus.values):
        n = int(round(v / 100 * N))
        ax.text(v + 1.2, yi, f"{pct(v)}  ({br(n)})", va="center",
                fontsize=9, color=INK_2)
    strip_axes(ax, keep_y_grid=False, keep_x_grid=True)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PRIMARY),
               plt.Rectangle((0, 0), 1, 1, fc=WARNING),
               plt.Rectangle((0, 0), 1, 1, fc=CRITICAL)]
    ax.legend(handles, ["ausência baixa (< 10%)",
                        "ausência moderada (10–50%)",
                        "ausência alta (> 50%)"], loc="lower right")
    ax.set_title("Panorama de valores ausentes — todas as 21 variáveis\n",
                 loc="left", fontsize=15, fontweight="bold", pad=24)
    ax.text(0.0, 1.012, "Ordenado pela taxa de ausência — maior no topo.",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")
    figure_caption(fig, NOTA)
    fig.savefig(os.path.join(OUT_FIG, "00_panorama_ausentes.png"),
                bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# EXECUCAO
# =========================================================================
def main():
    resumo = []
    print(f"Registros: {N} | Variáveis: {df.shape[1]}")

    # integridade do identificador
    dup = int(df["ID"].duplicated().sum())
    print(f"ID: {df['ID'].nunique()} valores distintos, {dup} duplicados "
          f"-> {'OK' if dup == 0 else 'PROBLEMA'} (não plotado: identificador)")

    plot_ausentes()
    print("  [ok] 00_panorama_ausentes.png")

    for col, (faixa, un) in QUANT.items():
        resumo.append(plot_quantitativa(col, faixa, un))
        print(f"  [ok] quant_{slug(col)}.png")

    for col in CATEG:
        ordem = ORDEM_PPA if col == "PPA" else None
        resumo.append(plot_categorica(col, ordem=ordem))
        print(f"  [ok] cat_{slug(col)}.png")

    resumo.append(plot_convenio())
    print("  [ok] cat_convenio.png")

    resumo.extend(plot_temporais())
    print("  [ok] temporal_datas.png")

    pd.DataFrame(resumo).to_csv(
        os.path.join(OUT_TAB, "resumo_univariado.csv"),
        index=False, encoding="utf-8-sig", sep=";", decimal=",")

    # tabela de frequencias completa das categoricas
    linhas = []
    for col in CATEG + ["Convenio"]:
        vc = df[col].value_counts(dropna=False)
        for cat, n in vc.items():
            linhas.append({"variavel": col,
                           "categoria": "(ausente)" if pd.isna(cat) else cat,
                           "frequencia": int(n),
                           "percentual": round(n / N * 100, 2)})
    pd.DataFrame(linhas).to_csv(
        os.path.join(OUT_TAB, "frequencias_categoricas.csv"),
        index=False, encoding="utf-8-sig", sep=";", decimal=",")

    print(f"\nFiguras -> {OUT_FIG}")
    print(f"Tabelas -> {OUT_TAB}")


if __name__ == "__main__":
    main()
