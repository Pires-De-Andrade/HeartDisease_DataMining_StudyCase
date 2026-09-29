# -*- coding: utf-8 -*-
"""
MODULO 4 - TRANSFORMACOES
Estudo de caso: condicao cardiaca em criancas e adolescentes

Entrada : data/processed/UCMF_clean.csv, gerado por 02_limpeza.py
          (contrato em docs/contrato-dados.md).

Duas camadas, separadas pela regra "modelar no treino, aplicar no teste":

  A. Deterministica (tabelas clinicas fixas, sem estatistica da amostra)
     -> aplicada a base inteira -> data/processed/UCMF_features.csv
        construcao: faixa_etaria, imc_classe, pa_classe, fc_classe,
                    sopro_presente, n_queixas
        eliminacao: id, convenio, hda2, ppa, imc, atendimento, dn

  B. Estatistica (imputacao, codificacao, escala)
     -> ColumnTransformer ajustado SOMENTE no treino (80/20 estratificado,
        random_state=42) e serializado -> data/processed/preprocessor.joblib
        + X_train.csv / X_test.csv / split_indices.csv
     Ficam fora da matriz de modelagem, mas dentro do UCMF_features.csv
     (SO_ANALISE): pa_sistolica e pa_diastolica (> 50% ausentes, efeito
     desprezivel na bivariada), sopro_presente (redundante com o one-hot de
     sopro) e n_queixas (sem associacao com o alvo).

Saidas para o relatorio: output/tabelas/transformacoes.csv (tabela-sintese),
output/tabelas/cobertura.csv, output/tabelas/validacao_ppa.csv (PPA original x
pa_classe reconstruida) e output/transform/*.png.
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clinico  # noqa: E402
from transformadores import ImputadorPorFaixa  # noqa: E402
from viz_theme import (apply_theme, strip_axes, figure_caption,   # noqa: E402
                       PRIMARY, ABSENT, INK, INK_2)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data", "processed", "UCMF_clean.csv")
OUT_DATA = os.path.join(BASE, "data", "processed")
OUT_FIG = os.path.join(BASE, "output", "transform")
OUT_TAB = os.path.join(BASE, "output", "tabelas")
for d in (OUT_DATA, OUT_FIG, OUT_TAB):
    os.makedirs(d, exist_ok=True)

apply_theme()
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ALVO = "normal_x_anormal"
SEED = 42
TESTE = 0.2
# rotulos da PPA original -> niveis da pa_classe (so para validacao)
MAPA_PPA = {"normal": "normal",
            "pre-hipertensão pas": "pre_has", "pre-hipertensão pad": "pre_has",
            "has-1 pas": "has1", "has-1 pad": "has1",
            "has-2 pas": "has2", "has-2 pad": "has2"}

# -------------------------------------------------------------------------
# Especificacao unica das transformacoes: gera a tabela-sintese do relatorio
# e dirige o pipeline. Mudou a decisao, muda-se aqui.
# -------------------------------------------------------------------------
ELIMINAR = {
    "id":          "identificador; sem poder preditivo",
    "convenio":    "sem relacao com o diagnostico; 28% ausente; 236 grafias apos a limpeza",
    "hda2":        "97% ausente (acima do limite de 50%)",
    "ppa":         "82% inutilizavel na origem; substituida por pa_classe reconstruida",
    "imc":         "derivado de peso e altura (redundante); mantida a classe por percentil",
    "atendimento": "usada apenas para recalcular a idade",
    "dn":          "usada apenas para recalcular a idade",
}
# ficam no UCMF_features.csv (analise), mas fora da matriz de modelagem
SO_ANALISE = {
    "pa_sistolica":  "56% ausente (> 50%); efeito desprezivel na bivariada; pa_classe carrega a informacao",
    "pa_diastolica": "56% ausente (> 50%); efeito desprezivel na bivariada; pa_classe carrega a informacao",
    "sopro_presente": "redundante com a coluna sopro_ausente do one-hot (colinearidade perfeita)",
    "n_queixas":     "sem associacao com o alvo na bivariada (p = 0,25)",
}
FAIXA = "faixa_etaria"
NUMERICAS_LOG = ["peso", "fc"]           # assimetria a direita; imputadas pela mediana da faixa etaria
NUMERICAS_FAIXA = ["altura"]             # imputada pela mediana da faixa etaria
NUMERICAS = ["idade"]                    # imputada pela mediana global
ORDINAIS = {FAIXA: clinico.ORDEM_FAIXA, "imc_classe": clinico.ORDEM_IMC,
            "pa_classe": clinico.ORDEM_PA, "fc_classe": clinico.ORDEM_FC}
NOMINAIS = ["sexo", "pulsos", "b2", "sopro", "hda1", "motivo1", "motivo2"]
MIN_FREQ = 30                            # agrega categorias raras

SINTESE = [
    ("id, convenio, hda2, ppa, imc, atendimento, dn", "Reducao", "Eliminacao manual",
     "irrelevantes ao diagnostico, > 50% ausentes, derivados com erro ou ja utilizados"),
    (", ".join(SO_ANALISE), "Reducao", "mantidas na base analitica; fora da matriz de modelagem",
     "PAS/PAD > 50% ausentes e sem efeito; sopro_presente redundante com o one-hot; n_queixas sem associacao"),
    ("faixa_etaria", "Criacao + discretizacao", "faixas clinicas (lactente, pre-escolar, escolar, adolescente)",
     "fisiologia e valores de referencia mudam por faixa; cortes definidos pelo usuario"),
    ("imc_classe", "Criacao + discretizacao", "percentil de IMC para idade e sexo (CDC): p5 / p85 / p95",
     "IMC absoluto nao tem significado sem idade e sexo; obesidade e fator de risco"),
    ("pa_classe", "Criacao + discretizacao", "reconstrucao da PPA: NHBPEP 2004 por sexo x idade x percentil de altura, criterio PAS ou PAD",
     "PPA original 82% inutilizavel; mesma tabela clinica da origem; associacao fraca com o alvo, mantida pela interpretabilidade"),
    ("fc_classe", "Criacao + discretizacao", "faixas de FC de repouso por idade (MedlinePlus/AHA)",
     "FC normal varia de 60 a 190 bpm conforme a idade"),
    ("sopro_presente, n_queixas", "Criacao", "binario / contagem",
     "construidos para a analise; nao entram na modelagem (ver Reducao)"),
    (ALVO, "Codificacao", "label: Normal = 0, Anormal = 1", "alvo binario; positivo = doenca"),
    (", ".join(ORDINAIS), "Codificacao", "ordinal (nao avaliavel = -1)", "ordem clinica real; preserva monotonicidade"),
    (", ".join(NOMINAIS), "Codificacao", f"one-hot (categorias com n < {MIN_FREQ} agregadas)",
     "nominais sem ordem e de baixa cardinalidade"),
    (", ".join(NUMERICAS_LOG), "Numerica", "imputacao pela mediana da faixa etaria -> log1p -> z-score",
     "assimetria a direita; valor tipico depende da idade"),
    (", ".join(NUMERICAS_FAIXA), "Numerica", "imputacao pela mediana da faixa etaria -> z-score",
     "31% ausente; mediana global daria 99 cm a um lactente"),
    (", ".join(NUMERICAS), "Numerica", "imputacao pela mediana -> z-score",
     "escala distinta; algoritmos baseados em distancia ou gradiente"),
    (ALVO, "Balanceamento", "sem reamostragem; class_weight e metricas por classe na modelagem",
     "desbalanceamento leve (57/43); classe minoritaria e a clinicamente importante"),
    ("todas as estatisticas", "Regra", "ajuste no treino (80/20 estratificado, seed 42), aplicacao no teste",
     "evitar vazamento de informacao do teste para o treino"),
]


# =========================================================================
# CAMADA A - deterministica
# =========================================================================
def carregar():
    if not os.path.exists(CLEAN):
        sys.exit(f"{CLEAN} nao existe: rode scripts/02_limpeza.py antes.")
    df = pd.read_csv(CLEAN)
    print(f"Entrada: {os.path.relpath(CLEAN, BASE)} ({len(df)} registros, "
          f"{df[ALVO].eq('Anormal').mean():.1%} Anormal)")
    return df


def construir_atributos(df):
    df = df.copy()
    meses = df["idade"] * 12
    df["faixa_etaria"] = df["idade"].map(clinico.faixa_etaria)

    perc_imc = [clinico.percentil_imc(s, m, i) for s, m, i in zip(df["sexo"], meses, df["imc"])]
    df["imc_classe"] = [clinico.classe_imc(p) for p in perc_imc]

    perc_alt = [clinico.percentil_altura(s, m, a) for s, m, a in zip(df["sexo"], meses, df["altura"])]
    df["pa_classe"] = [clinico.classe_pa(s, i, pas, pad, pa)
                       for s, i, pas, pad, pa in zip(df["sexo"], df["idade"], df["pa_sistolica"],
                                                     df["pa_diastolica"], perc_alt)]
    df["fc_classe"] = [clinico.classe_fc(i, f) for i, f in zip(df["idade"], df["fc"])]

    sopro = df["sopro"].str.strip().str.lower()
    hda1 = df["hda1"].str.strip().str.lower()
    df["sopro_presente"] = (sopro.notna() & (sopro != "ausente")).astype(int)
    df["n_queixas"] = ((hda1.notna() & (hda1 != "assintomático")).astype(int)
                       + df["hda2"].notna().astype(int))
    df[ALVO] = (df[ALVO] == "Anormal").astype(int)
    return df


def cobertura(df):
    linhas = []
    for col in ORDINAIS:
        n = int((df[col].notna() & (df[col] != clinico.NAO_AVALIAVEL)).sum())
        linhas.append({"atributo": col, "n_avaliado": n, "pct_avaliado": round(100 * n / len(df), 1)})
    return pd.DataFrame(linhas)


def validar_ppa(df):
    """Concordancia entre a PPA original (onde existe) e a pa_classe reconstruida."""
    orig = df["ppa"].str.strip().str.lower().map(MAPA_PPA)
    m = orig.notna() & (df["pa_classe"] != clinico.NAO_AVALIAVEL)
    tab = pd.crosstab(orig[m].rename("ppa_original"), df.loc[m, "pa_classe"]
                      ).reindex(index=clinico.ORDEM_PA, columns=clinico.ORDEM_PA, fill_value=0)
    concordancia = float((orig[m] == df.loc[m, "pa_classe"]).mean())
    print(f"Validação PPA: {int(orig.notna().sum())} rótulos originais, {int(m.sum())} avaliados "
          f"pela reconstrução, concordância exata {concordancia:.1%}")
    return tab, concordancia


# =========================================================================
# CAMADA B - estatistica (ajustada so no treino)
# =========================================================================
def montar_preprocessador():
    num_log = Pipeline([("imputar", ImputadorPorFaixa(grupo=FAIXA)),
                        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("escalar", StandardScaler())])
    num_faixa = Pipeline([("imputar", ImputadorPorFaixa(grupo=FAIXA)),
                          ("escalar", StandardScaler())])
    num = Pipeline([("imputar", SimpleImputer(strategy="median")),
                    ("escalar", StandardScaler())])
    ordinal = OrdinalEncoder(categories=list(ORDINAIS.values()),
                             handle_unknown="use_encoded_value", unknown_value=-1,
                             encoded_missing_value=-1)
    nominal = Pipeline([("imputar", SimpleImputer(strategy="constant", fill_value="nao_informado")),
                        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                 min_frequency=MIN_FREQ, sparse_output=False))])
    return ColumnTransformer([
        ("num_log", num_log, NUMERICAS_LOG + [FAIXA]),
        ("num_faixa", num_faixa, NUMERICAS_FAIXA + [FAIXA]),
        ("num", num, NUMERICAS),
        ("ord", ordinal, list(ORDINAIS)),
        ("nom", nominal, NOMINAIS),
    ], verbose_feature_names_out=False)


# =========================================================================
# FIGURAS
# =========================================================================
def fig_escala(treino):
    """Peso: bruto -> log1p -> z-score (parametros do treino)."""
    v = treino["peso"].dropna()
    etapas = [("A · peso limpo (kg)", v),
              ("B · log1p(peso)", np.log1p(v)),
              ("C · z-score de log1p(peso)", (np.log1p(v) - np.log1p(v).mean()) / np.log1p(v).std())]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (titulo, s) in zip(axes, etapas):
        ax.hist(s, bins=40, color=PRIMARY, edgecolor="#fcfcfb", linewidth=0.5)
        ax.set_title(titulo, loc="left", pad=8)
        ax.set_ylabel("Frequência")
        strip_axes(ax)
        ax.text(0.98, 0.95, f"assimetria = {s.skew():.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=INK_2)
    fig.suptitle("Transformação numérica — log reduz a assimetria; z-score coloca na escala comum",
                 x=0.0, ha="left", fontsize=14, fontweight="bold", y=1.04)
    figure_caption(fig, f"Conjunto de treino (n = {len(v):,}). Média e desvio calculados "
                        "apenas no treino e reaplicados ao teste.".replace(",", "."))
    fig.savefig(os.path.join(OUT_FIG, "transform_peso_log_zscore.png"), bbox_inches="tight")
    plt.close(fig)


def fig_classes(df):
    """Proporcao de Anormal em cada classe clinica construida."""
    cols = ["faixa_etaria", "imc_classe", "pa_classe", "fc_classe"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw={"wspace": 0.75})
    for ax, col in zip(axes, cols):
        ordem = ORDINAIS[col] + [clinico.NAO_AVALIAVEL]
        g = df.groupby(col)[ALVO].agg(["mean", "size"]).reindex(ordem).dropna()
        y = np.arange(len(g))[::-1]
        cores = [ABSENT if i == clinico.NAO_AVALIAVEL else PRIMARY for i in g.index]
        ax.barh(y, g["mean"] * 100, color=cores, height=0.66)
        ax.set_yticks(y)
        ax.set_yticklabels(g.index, fontsize=9)
        ax.set_xlim(0, 85)
        ax.set_xlabel("% Anormal")
        for yi, (m, n) in zip(y, zip(g["mean"], g["size"])):
            ax.text(m * 100 + 1.5, yi, f"{m*100:.0f}%  (n={int(n):,})".replace(",", "."),
                    va="center", fontsize=8.5, color=INK_2)
        ax.axvline(df[ALVO].mean() * 100, color=INK, linestyle="--", linewidth=1)
        ax.set_title(col, loc="left", pad=8)
        strip_axes(ax, keep_y_grid=False, keep_x_grid=True)
    fig.suptitle("Atributos construídos × alvo — linha tracejada = taxa global de Anormal",
                 x=0.0, ha="left", fontsize=14, fontweight="bold", y=1.04)
    figure_caption(fig, "Base inteira (transformações determinísticas). Cinza = não avaliável "
                        "(falta de PA, altura, idade ou sexo).")
    fig.savefig(os.path.join(OUT_FIG, "transform_classes_por_alvo.png"), bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# EXECUCAO
# =========================================================================
def main():
    df = carregar()

    # --- camada A ---------------------------------------------------------
    df = construir_atributos(df)
    cob = cobertura(df)
    cob.to_csv(os.path.join(OUT_TAB, "cobertura.csv"), index=False, sep=";", decimal=",")
    print("Cobertura dos atributos construídos:\n" + cob.to_string(index=False))
    tab_ppa, _ = validar_ppa(df)
    tab_ppa.to_csv(os.path.join(OUT_TAB, "validacao_ppa.csv"), sep=";")

    ids = df["id"]
    features = df.drop(columns=list(ELIMINAR))
    features.insert(0, "id", ids)
    features.to_csv(os.path.join(OUT_DATA, "UCMF_features.csv"), index=False)
    print(f"[ok] UCMF_features.csv: {features.shape[0]} x {features.shape[1]}")

    # --- camada B ---------------------------------------------------------
    X = features.drop(columns=["id", ALVO] + list(SO_ANALISE))
    y = features[ALVO]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TESTE, stratify=y, random_state=SEED)

    pre = montar_preprocessador()
    pre.fit(X_tr)                                   # parametros SO do treino
    nomes = pre.get_feature_names_out()
    Xt_tr = pd.DataFrame(pre.transform(X_tr), columns=nomes, index=X_tr.index)
    Xt_te = pd.DataFrame(pre.transform(X_te), columns=nomes, index=X_te.index)

    joblib.dump(pre, os.path.join(OUT_DATA, "preprocessor.joblib"))
    Xt_tr.assign(alvo=y_tr).to_csv(os.path.join(OUT_DATA, "X_train.csv"), index=False)
    Xt_te.assign(alvo=y_te).to_csv(os.path.join(OUT_DATA, "X_test.csv"), index=False)
    pd.DataFrame({"id": ids.loc[X.index],
                  "conjunto": np.where(X.index.isin(X_tr.index), "treino", "teste")}
                 ).to_csv(os.path.join(OUT_DATA, "split_indices.csv"), index=False)
    print(f"[ok] treino {Xt_tr.shape}, teste {Xt_te.shape}; "
          f"alvo treino {y_tr.mean():.1%} / teste {y_te.mean():.1%} Anormal")

    # --- saidas do relatorio -----------------------------------------------
    pd.DataFrame(SINTESE, columns=["atributo", "frente", "transformacao", "justificativa"]).to_csv(
        os.path.join(OUT_TAB, "transformacoes.csv"), index=False, sep=";", encoding="utf-8-sig")
    fig_escala(X_tr)
    fig_classes(df)
    print(f"[ok] tabelas -> {OUT_TAB} | figuras -> {OUT_FIG} | dados -> {OUT_DATA}")


if __name__ == "__main__":
    main()
