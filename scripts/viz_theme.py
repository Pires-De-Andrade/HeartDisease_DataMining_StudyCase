# -*- coding: utf-8 -*-
"""
Tema visual compartilhado pelos modulos do processo KDD.

Paleta validada (light surface #fcfcfb): as cores categoricas sao atribuidas
em ordem fixa, nunca cicladas. Cores de status (critical/warning) sao
reservadas para sinalizar problemas de qualidade e nunca representam uma serie.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- superficies e tinta -------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- paleta categorica (ordem fixa) --------------------------------------
SERIES = [
    "#2a78d6",  # 1 azul
    "#eb6834",  # 2 laranja
    "#1baf7a",  # 3 agua
    "#eda100",  # 4 amarelo
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 verde
    "#4a3aa7",  # 7 violeta
    "#e34948",  # 8 vermelho
]
PRIMARY = SERIES[0]

# --- status (reservado para problemas de qualidade) ----------------------
CRITICAL = "#d03b3b"   # valores invalidos / impossiveis
WARNING = "#fab219"    # valores suspeitos / fora de faixa clinica
GOOD = "#0ca30c"
ABSENT = "#b8b6ae"     # categoria "(ausente)"

FONTS = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def apply_theme():
    """Aplica o tema global do matplotlib. Chamar uma vez por script."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": FONTS,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def strip_axes(ax, keep_y_grid=True, keep_x_grid=False):
    """Remove cromo desnecessario: bordas superiores/direitas e grade redundante."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.yaxis.grid(keep_y_grid)
    ax.xaxis.grid(keep_x_grid)


def figure_caption(fig, text):
    """Legenda de rodape com a fonte dos dados / nota metodologica."""
    fig.text(0.0, -0.015, text, ha="left", va="top",
             fontsize=8, color=MUTED, wrap=True)
