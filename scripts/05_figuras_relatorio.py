# -*- coding: utf-8 -*-
"""
FIGURAS DO RELATORIO FINAL

Gera as 4 figuras do relatorio de pre-processamento ja no tamanho de impressao
(largura util de 17 cm, escala 1:1 no documento), com rotulos por extenso -
o relatorio nao pode conter nada que pareca codigo. Titulos e legendas das
figuras ficam no documento; aqui so os paineis.

  fig1_univariada.png   - 7 quantitativas brutas + alvo bruto (item 1)
  fig2_peso_altura.png  - peso x altura antes/depois da limpeza (itens 4-5)
  fig3_efeitos.png      - ranking do tamanho de efeito contra o alvo (item 6)
  fig4_quant_alvo.png   - idade, peso, FC e PAS por grupo (item 6)

A classificacao dos pares invalidos da fig2 usa as mesmas funcoes de
qualidade.py que 02_limpeza.py usa, para que os numeros coincidam.

Saida: output/relatorio/*.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qualidade import (BASE, MAPA_SEXO, carregar_bruto, parse_data, corrigir_seculo,
                       idade_por_datas, fora_faixa, avaliar_antropometria)
from viz_theme import (apply_theme, strip_axes, PRIMARY, CRITICAL, WARNING,
                       ABSENT, INK, INK_2, MUTED, SURFACE, SERIES)

OUT = os.path.join(BASE, "output", "relatorio")
os.makedirs(OUT, exist_ok=True)
CLEAN = os.path.join(BASE, "data", "processed", "UCMF_clean.csv")
BIV = os.path.join(BASE, "output", "tabelas", "resumo_bivariado.csv")

LARGURA = 6.69          # 17 cm em polegadas

apply_theme()
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 7,
    "savefig.dpi": 300, "axes.linewidth": 0.6, "grid.linewidth": 0.5,
    # fundo branco puro: o off-white da tela vira uma "caixa" no papel
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
SURFACE = "white"

COR_GRUPO = {"Normal": SERIES[0], "Anormal": SERIES[1]}


def br(n):
    return f"{int(n):,}".replace(",", ".")


def num(x, casas=1):
    s = f"{x:,.{casas}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


# =========================================================================
# FIGURA 1 - univariada: 7 quantitativas brutas + alvo bruto
# =========================================================================
QUANT = [
    # coluna,          rotulo,               faixa,       resumo extra
    ("Peso",          "Peso (kg)",          (0.5, 150), ["zeros", "min"]),
    ("Altura",        "Altura (cm)",        (30, 200),  ["zeros"]),
    ("IMC",           "IMC (kg/m²)",        (8, 45),    ["max"]),
    ("IDADE",         "Idade (anos)",       (0, 19),    ["neg", "max"]),
    ("PA SISTOLICA",  "PA sistólica (mmHg)", (50, 200), ["max"]),
    ("PA DIASTOLICA", "PA diastólica (mmHg)", (20, 150), ["min"]),
    ("FC",            "FC (bpm)",           (40, 220),  ["texto", "max"]),
]


def fig1():
    bruto = carregar_bruto()
    n = len(bruto)
    fig, axes = plt.subplots(2, 4, figsize=(LARGURA, 3.2))
    axes = axes.ravel()

    for ax, (col, rotulo, (lo, hi), extras) in zip(axes, QUANT):
        s = bruto[col]
        v = pd.to_numeric(s, errors="coerce")
        n_txt = int((s.notna() & v.isna()).sum())
        v = v.dropna()
        _, bins, patches = ax.hist(v, bins=45, color=PRIMARY,
                                   edgecolor=SURFACE, linewidth=0.3)
        for p, b0, b1 in zip(patches, bins[:-1], bins[1:]):
            if (b0 + b1) / 2 < lo or (b0 + b1) / 2 > hi:
                p.set_facecolor(CRITICAL)
        ax.set_yscale("log")
        ax.set_title(rotulo, loc="left", pad=3, fontweight="semibold")

        partes = [f"ausentes {num(s.isna().mean() * 100)}%"]
        for e in extras:
            if e == "zeros":
                partes.append(f"{br((v == 0).sum())} zeros")
            elif e == "neg":
                partes.append(f"{br((v < 0).sum())} negativos")
            elif e == "min":
                partes.append(f"mín. {num(v.min(), 0)}")
            elif e == "max":
                partes.append(f"máx. {num(v.max(), 0)}")
            elif e == "texto":
                partes.append(f"{n_txt} textos")
        ax.set_xlabel(" · ".join(partes), fontsize=6.6, color=INK_2, labelpad=2)
        strip_axes(ax)
        ax.tick_params(length=2, pad=1.5)

    # painel 8: o alvo bruto
    ax = axes[7]
    alvo = bruto["NORMAL X ANORMAL"].fillna("(ausente)").value_counts()
    ordem = ["Normal", "Anormal", "anormal", "Normais", "(ausente)"]
    alvo = alvo.reindex(ordem)
    y = np.arange(len(ordem))[::-1]
    cores = [PRIMARY, PRIMARY, CRITICAL, CRITICAL, ABSENT]
    ax.barh(y, alvo.values, color=cores, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(ordem, fontsize=6.8)
    ax.set_xlim(0, alvo.max() * 1.55)
    for yi, val in zip(y, alvo.values):
        ax.text(val + alvo.max() * 0.03, yi, br(val), va="center",
                fontsize=6.5, color=INK_2)
    ax.set_title("Alvo: NORMAL X ANORMAL", loc="left", pad=3, fontweight="semibold")
    ax.set_xlabel(f"4 rótulos · ausentes {num(bruto['NORMAL X ANORMAL'].isna().mean() * 100)}%",
                  fontsize=6.6, color=INK_2, labelpad=2)
    strip_axes(ax, keep_y_grid=False, keep_x_grid=True)
    ax.tick_params(length=2, pad=1.5)
    ax.set_xticks([0, 3000, 6000])
    ax.set_xticklabels(["0", "3.000", "6.000"])

    for a in (axes[0], axes[4]):
        a.set_ylabel("frequência (log)", fontsize=6.8)

    h = [plt.Rectangle((0, 0), 1, 1, fc=PRIMARY), plt.Rectangle((0, 0), 1, 1, fc=CRITICAL),
         plt.Rectangle((0, 0), 1, 1, fc=ABSENT)]
    fig.legend(h, ["dentro da faixa plausível / rótulo válido",
                   "fora da faixa plausível / rótulo fora do padrão",
                   "ausente"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.035),
               handlelength=1.2, columnspacing=1.6)
    fig.tight_layout(h_pad=1.0, w_pad=0.6, rect=(0, 0.04, 1, 1))
    out = os.path.join(OUT, "fig1_univariada.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out


# =========================================================================
# FIGURA 2 - peso x altura antes e depois (mesma logica de 02_limpeza.py)
# =========================================================================
def fig2():
    bruto = carregar_bruto()
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

    limpo = pd.read_csv(CLEAN)
    lp = limpo[["peso", "altura", "imc"]].dropna(subset=["peso", "altura"])
    sem_imc = lp["imc"].isna()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(LARGURA, 2.5),
                                   sharex=True, sharey=True)
    axA.scatter(a[valido], p[valido], s=2, color=PRIMARY, alpha=0.35,
                linewidths=0, label=f"par plausível ({br(valido.sum())})")
    axA.scatter(a[invalido], p[invalido], s=7, color=CRITICAL, marker="x",
                linewidths=0.5, label=f"par inválido ou implausível\npara a idade ({br(invalido.sum())})")
    axA.annotate(f"{br(zerados)} registros com\npeso = 0 e altura = 0",
                 xy=(0, 0), xytext=(18, 118), textcoords="data", fontsize=6.6,
                 color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=0.6))
    axA.set_title(f"A · Dados brutos ({br(tem_par.sum())} pares)", loc="left", pad=3)

    axB.scatter(lp.loc[~sem_imc, "altura"], lp.loc[~sem_imc, "peso"], s=2,
                color=PRIMARY, alpha=0.35, linewidths=0,
                label=f"par coerente ({br((~sem_imc).sum())})")
    axB.scatter(lp.loc[sem_imc, "altura"], lp.loc[sem_imc, "peso"], s=9, marker="^",
                facecolors="none", edgecolors=WARNING, linewidths=0.6,
                label=f"par incoerente mantido,\nIMC anulado ({br(sem_imc.sum())})")
    axB.set_title(f"B · Após a limpeza ({br(len(lp))} pares)", loc="left", pad=3)

    for ax in (axA, axB):
        ax.set_ylim(-50, 205)
        ax.set_xlabel("Altura (cm)")
        leg = ax.legend(loc="upper right", markerscale=2.2, handletextpad=0.3,
                        borderaxespad=0.2)
        for hnd in leg.legend_handles:
            hnd.set_alpha(1)
        strip_axes(ax, keep_y_grid=True, keep_x_grid=True)
        ax.tick_params(length=2, pad=1.5)
    axA.set_ylabel("Peso (kg)")
    fig.tight_layout(w_pad=1.2)
    out = os.path.join(OUT, "fig2_peso_altura.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out


# =========================================================================
# FIGURA 3 - ranking do tamanho de efeito contra o alvo
# =========================================================================
ROTULO_BIV = {
    "sopro": "Sopro", "motivo2": "Motivo 2", "motivo1": "Motivo 1",
    "peso": "Peso", "altura": "Altura", "fc": "Frequência cardíaca",
    "idade": "Idade", "b2": "B2 (segunda bulha)", "imc": "IMC",
    "hda1": "HDA 1 (queixa principal)", "pulsos": "Pulsos",
    "pa_diastolica": "PA diastólica", "pa_sistolica": "PA sistólica", "sexo": "Sexo",
}


def fig3():
    t = pd.read_csv(BIV, sep=";", decimal=",", encoding="utf-8-sig")
    t = t[t["variavel"].isin(ROTULO_BIV)].copy()
    t["abs"] = t["efeito"].abs()
    t = t.sort_values("abs")
    quant = t["tipo"].eq("quantitativa")

    fig, ax = plt.subplots(figsize=(LARGURA, 2.45))
    y = np.arange(len(t))
    cores = [SERIES[1] if q else SERIES[0] for q in quant]
    ax.barh(y, t["abs"], color=cores, height=0.62)

    rotulos = []
    for _, r in t.iterrows():
        nome = ROTULO_BIV[r["variavel"]]
        if r["tipo"] == "quantitativa":
            sentido = "maior" if r["mediana_anormal"] > r["mediana_normal"] else "menor"
            if r["mediana_anormal"] == r["mediana_normal"]:
                sentido = "igual"
            nome += f" ({sentido} no Anormal)"
        rotulos.append(nome)
    ax.set_yticks(y)
    ax.set_yticklabels(rotulos, fontsize=6.9, color=INK_2)

    for yi, (_, r) in zip(y, t.iterrows()):
        v = r["abs"]
        # 0,095 arredondado para 0,10 pareceria "fraca": mostra 3 casas nesses casos
        casas = 3 if any(round(v, 2) == c and v < c for c in (0.1, 0.3, 0.5)) else 2
        txt = num(v, casas)
        if r["p_valor"] >= 0.05:
            txt += "  (não significativo)"
        ax.text(v + 0.008, yi, txt, va="center", fontsize=6.5, color=INK_2, zorder=5,
                bbox=dict(boxstyle="square,pad=0.12", fc=SURFACE, ec="none"))

    # faixas de Cohen
    for corte in (0.1, 0.3, 0.5):
        ax.axvline(corte, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
    topo = len(t) - 0.35
    for x0, x1, nome in [(0, 0.1, "desprezível"), (0.1, 0.3, "fraca"),
                         (0.3, 0.5, "moderada"), (0.5, 0.95, "forte")]:
        ax.text((x0 + x1) / 2, topo, nome, ha="center", va="bottom",
                fontsize=6.6, color=MUTED, style="italic")
    ax.set_xlim(0, 0.95)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: num(x, 1)))
    ax.set_ylim(-0.6, len(t) + 0.3)
    ax.set_xlabel("Tamanho de efeito (0 = nenhuma associação com o alvo)")
    strip_axes(ax, keep_y_grid=False, keep_x_grid=False)
    ax.tick_params(length=2, pad=1.5)

    h = [plt.Rectangle((0, 0), 1, 1, fc=SERIES[0]), plt.Rectangle((0, 0), 1, 1, fc=SERIES[1])]
    ax.legend(h, ["qualitativa: V de Cramér (qui-quadrado)",
                  "quantitativa: |r| rank-biserial (Mann-Whitney U)"],
              loc="lower right", handlelength=1.2)
    fig.tight_layout()
    out = os.path.join(OUT, "fig3_efeitos.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out


# =========================================================================
# FIGURA 4 - idade, peso, FC e PAS por grupo
# =========================================================================
def fig4():
    df = pd.read_csv(CLEAN)
    vars4 = [("idade", "Idade (anos)"), ("peso", "Peso (kg)"),
             ("fc", "Frequência cardíaca (bpm)"), ("pa_sistolica", "PA sistólica (mmHg)")]
    fig, axes = plt.subplots(1, 4, figsize=(LARGURA, 1.95))
    for ax, (col, rotulo) in zip(axes, vars4):
        dados, ns = [], []
        for g in ("Normal", "Anormal"):
            v = df.loc[df["normal_x_anormal"] == g, col].dropna()
            dados.append(v)
            ns.append(len(v))
        bp = ax.boxplot(dados, widths=0.55, patch_artist=True,
                        medianprops=dict(color=INK, linewidth=1.1),
                        whiskerprops=dict(color=INK_2, linewidth=0.6),
                        capprops=dict(color=INK_2, linewidth=0.6),
                        flierprops=dict(marker="o", markersize=1.2, markerfacecolor=MUTED,
                                        markeredgecolor="none", alpha=0.35))
        for patch, g in zip(bp["boxes"], ("Normal", "Anormal")):
            patch.set_facecolor(COR_GRUPO[g])
            patch.set_alpha(0.55)
            patch.set_edgecolor(COR_GRUPO[g])
            patch.set_linewidth(0.8)
        for i, v in enumerate(dados, start=1):
            ax.text(i + 0.33, v.median(), num(v.median(), 1 if col in ("idade", "peso") else 0),
                    va="center", ha="left", fontsize=6.3, color=INK,
                    fontweight="semibold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Normal\nn={br(ns[0])}", f"Anormal\nn={br(ns[1])}"],
                           fontsize=6.4)
        ax.set_title(rotulo, loc="left", pad=3, fontsize=7.5, fontweight="semibold")
        ax.set_xlim(0.5, 2.75)
        strip_axes(ax)
        ax.tick_params(length=2, pad=1.5)
    fig.tight_layout(w_pad=0.8)
    out = os.path.join(OUT, "fig4_quant_alvo.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        print("[ok]", f())
