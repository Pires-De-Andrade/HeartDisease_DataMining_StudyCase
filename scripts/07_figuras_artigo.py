# -*- coding: utf-8 -*-
"""
FIGURAS DO ARTIGO (A2)

Gera as figuras do artigo IEEE em PDF vetorial, no tamanho de impressao
(duas colunas = 7,16 in), direto em paper/figures/.

  fig_dados.pdf - (a) % Anormal por achado da ausculta e pelo motivo "sopro"
                  relatado no encaminhamento (vazamento de dados);
                  (b) prevalencia anual de Anormal, 2003-2010 (deriva temporal).

Populacao: a principal do artigo (docs/proposta.md, secao 5) - base limpa
sem MOTIVO1 = "1 - Cardiopatia ja estabelecida".
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_theme import apply_theme, strip_axes, PRIMARY, INK, INK_2, MUTED  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data", "processed", "UCMF_clean.csv")
OUT = os.path.join(BASE, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

LARGURA = 7.16            # \textwidth do IEEEtran (duas colunas), em polegadas
EXCLUIR_MOTIVO1 = "1 - Cardiopatia já estabelecida"

apply_theme()
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,
})


def fmt(n):
    return f"{int(n):,}".replace(",", ".")


def populacao():
    df = pd.read_csv(CLEAN)
    df = df[df["motivo1"] != EXCLUIR_MOTIVO1].copy()
    df["y"] = (df["normal_x_anormal"] == "Anormal").astype(int)
    df["ano"] = pd.to_datetime(df["atendimento"], errors="coerce").dt.year
    return df


def painel_vazamento(ax, df):
    sopro = df["sopro"].str.strip().str.lower()
    b2 = df["b2"].str.strip().str.lower()
    grupos = [
        ("Sem sopro (ausculta)", sopro.eq("ausente")),
        ("Sopro sistólico (ausculta)", sopro.eq("sistólico")),
        ("B2 alterada (ausculta)", b2.notna() & b2.ne("normal")),
        ("Motivo “sopro” (encaminhamento)", df["motivo2"].eq("6 - Sopro")),
    ]
    taxas = [(rot, df.loc[m, "y"].mean() * 100, m.sum()) for rot, m in grupos]
    yy = np.arange(len(taxas))[::-1]
    ax.barh(yy, [t for _, t, _ in taxas], color=PRIMARY, height=0.6)
    for yi, (_, t, n) in zip(yy, taxas):
        ax.text(t + 1.5, yi, f"{t:.0f}%  (n = {fmt(n)})", va="center", fontsize=7, color=INK_2,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels([r for r, _, _ in taxas], color=INK)
    glob = df["y"].mean() * 100
    ax.axvline(glob, color=MUTED, linestyle="--", linewidth=0.9)
    ax.text(glob + 1, yy[0] - 0.5, f"global {glob:.0f}%", fontsize=6.5, color=MUTED, va="center")
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% Anormal")
    ax.set_title("(a) Achados da ausculta e motivo do encaminhamento", loc="left")
    strip_axes(ax, keep_y_grid=False, keep_x_grid=True)


def painel_tempo(ax, df):
    g = df[df["ano"].between(2003, 2010)].groupby("ano")["y"].agg(["mean", "size"])
    x, t = g.index.astype(int), g["mean"] * 100
    ax.plot(x, t, color=PRIMARY, linewidth=2, marker="o", markersize=4)
    for xi, ti, dx, dy in [(x[0], t.iloc[0], 0, 6), (x[-1], t.iloc[-1], -4, -12)]:
        ax.annotate(f"{ti:.0f}%", (xi, ti), textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=7, color=INK)
    ax.axvspan(2008.5, 2010.5, color=MUTED, alpha=0.12, linewidth=0)
    ax.text(2009.5, 58, "teste\ntemporal", ha="center", va="top", fontsize=6.5, color=INK_2)
    ax.set_xlim(2002.5, 2010.5)
    ax.set_ylim(0, 60)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{v % 100:02d}" for v in x])
    ax.set_xlabel("Ano do atendimento (20xx)")
    ax.set_ylabel("% Anormal")
    ax.set_title("(b) Prevalência anual de Anormal", loc="left")
    strip_axes(ax)


def main():
    df = populacao()
    fig, (a, b) = plt.subplots(1, 2, figsize=(LARGURA, 2.0),
                               gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.55})
    painel_vazamento(a, df)
    painel_tempo(b, df)
    destino = os.path.join(OUT, "fig_dados.pdf")
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(destino.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print(f"[ok] {os.path.relpath(destino, BASE)} (n = {fmt(len(df))}, "
          f"{df['y'].mean():.1%} Anormal)")


if __name__ == "__main__":
    main()
