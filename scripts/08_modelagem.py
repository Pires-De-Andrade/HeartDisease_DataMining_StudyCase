# -*- coding: utf-8 -*-
"""
MODULO 8 - MODELAGEM (A2). Protocolo pre-registrado: docs/proposta.md.

  --fase cv     so o treino: busca em grade, melhores hiperparametros e limiares
                (fixados antes de qualquer contato com o teste).
  --fase teste  exige este script commitado; refaz os ajustes (deterministicos),
                confere que batem com a fase cv e avalia o teste UMA vez:
                metricas com IC 95% por bootstrap, hipoteses H1-H4 e interpretacao.

Saidas: output/modelagem/
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clinico  # noqa: E402
from transformadores import ImputadorPorFaixa  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output", "modelagem")
SCRIPT = os.path.relpath(os.path.abspath(__file__), BASE)
SEED, N_BOOT, MIN_FREQ = 42, 1000, 30
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
ALVO = "normal_x_anormal"
SOPRO_ENCAMINHAMENTO = "6 - Sopro"

# --- etapas de informacao (cumulativas; proposta, secao 5) ---------------
E0 = {"num": ["idade"], "ord": ["faixa_etaria"], "nom": ["sexo", "motivo1", "motivo2", "hda1"]}
E1 = {"num": ["idade"], "num_log": ["peso", "fc"], "num_faixa": ["altura"],
      "ord": ["faixa_etaria", "imc_classe", "pa_classe", "fc_classe"],
      "nom": E0["nom"] + ["pulsos"]}
E2 = {**E1, "nom": E1["nom"] + ["sopro", "b2"]}
ETAPAS = {"idade": {"num": ["idade"]}, "E0": E0, "E1": E1, "E2": E2,
          "E0-": E0, "E1-": E1}          # "-" = ablacao do motivo "sopro" do encaminhamento
ORDENS = {"faixa_etaria": clinico.ORDEM_FAIXA, "imc_classe": clinico.ORDEM_IMC,
          "pa_classe": clinico.ORDEM_PA, "fc_classe": clinico.ORDEM_FC}

# --- modelos e grade (proposta, secao 6) ---------------------------------
MODELOS = {
    "prevalencia": (DummyClassifier(strategy="prior"), {}),
    "idade": (LogisticRegression(max_iter=5000), {}),
    "RL": (LogisticRegression(class_weight="balanced", max_iter=5000),
           {"clf__C": [0.001, 0.01, 0.1, 1, 10]}),
    "GB": (HistGradientBoostingClassifier(class_weight="balanced", max_iter=300,
                                          early_stopping=False, random_state=SEED),
           {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [15, 31],
            "clf__min_samples_leaf": [20, 80], "clf__l2_regularization": [0, 1]}),
}
PRINCIPAIS = [("prevalencia", "E0"), ("idade", "idade")] + \
             [(m, e) for e in ["E0", "E1", "E2", "E0-", "E1-"] for m in ["RL", "GB"]]
CONFIGS = {"aleatorio": PRINCIPAIS, "temporal": PRINCIPAIS,
           "aleatorio_2a19": [("RL", "E1"), ("GB", "E1")]}     # H2


def carregar():
    f = pd.read_csv(os.path.join(BASE, "data", "processed", "UCMF_features.csv"))
    c = pd.read_csv(os.path.join(BASE, "data", "processed", "UCMF_clean.csv"),
                    usecols=["id", "atendimento"])
    df = f.merge(c, on="id")
    df = df[df["motivo1"] != "1 - Cardiopatia já estabelecida"].reset_index(drop=True)
    df["ano"] = pd.to_datetime(df["atendimento"], errors="coerce").dt.year
    assert (len(df), df[ALVO].sum()) == (10311, 4084), "populacao difere da proposta"
    return df


def splits(df):
    tr, te = train_test_split(df.index, test_size=0.2, stratify=df[ALVO], random_state=SEED)
    adulto = df["idade"] >= 2
    s = {"aleatorio": (tr, te),
         "temporal": (df.index[df["ano"] <= 2008], df.index[df["ano"].between(2009, 2010)]),
         "aleatorio_2a19": (tr[adulto[tr]], te[adulto[te]])}
    assert (len(s["temporal"][0]), len(s["temporal"][1])) == (7520, 2715)
    for a, b in s.values():
        assert not set(a) & set(b)
    return s


def preparar(df, etapa):
    """Colunas da etapa; na ablacao, o motivo 'sopro' vira ausente ('nao_informado')."""
    cols = list(dict.fromkeys(sum(ETAPAS[etapa].values(), [])))
    X = df[cols].copy()
    if etapa.endswith("-"):
        X["motivo2"] = X["motivo2"].mask(X["motivo2"] == SOPRO_ENCAMINHAMENTO)
    return X


def preprocessador(etapa):
    """ColumnTransformer da A1 (04_transformacoes.py), restrito as colunas da etapa."""
    et = ETAPAS[etapa]
    blocos = {
        "num": Pipeline([("imp", SimpleImputer(strategy="median")), ("z", StandardScaler())]),
        "num_log": Pipeline([("imp", ImputadorPorFaixa()), ("log", FunctionTransformer(np.log1p)),
                             ("z", StandardScaler())]),
        "num_faixa": Pipeline([("imp", ImputadorPorFaixa()), ("z", StandardScaler())]),
        "ord": OrdinalEncoder(categories=[ORDENS[c] for c in et.get("ord", [])],
                              handle_unknown="use_encoded_value", unknown_value=-1,
                              encoded_missing_value=-1),
        "nom": Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="nao_informado")),
                         ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                              min_frequency=MIN_FREQ, sparse_output=False))]),
    }
    faixa = {"num_log", "num_faixa"}       # imputacao por faixa precisa da coluna de faixa
    return ColumnTransformer([(k, blocos[k], et[k] + (["faixa_etaria"] if k in faixa else []))
                              for k in et])


def limiares(y, p):
    """Fixados no treino (OOF): sens >= 0,80 (triagem) e espec >= 0,92 (comparacao)."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr >= 0.80)]), float(thr[np.flatnonzero(fpr <= 0.08)[-1]])


def ajustar(df, tr, modelo, etapa):
    clf, grade = MODELOS[modelo]
    pipe = Pipeline([("pre", preprocessador(etapa)), ("clf", clf)])
    X, y = preparar(df.loc[tr], etapa), df.loc[tr, ALVO]
    gs = GridSearchCV(pipe, grade, scoring="roc_auc", cv=CV, n_jobs=-1).fit(X, y)
    oof = cross_val_predict(gs.best_estimator_, X, y, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    return gs, limiares(y, oof)


def metricas(y, p, t_tri, t_comp):
    out = {"auc_roc": roc_auc_score(y, p), "auc_pr": average_precision_score(y, p),
           "brier": brier_score_loss(y, p)}
    for nome, t in [("triagem", t_tri), ("comp", t_comp)]:
        pos = p >= t
        out[f"sens_{nome}"] = pos[y == 1].mean()
        out[f"espec_{nome}"] = (~pos)[y == 0].mean()
    return out


def ic(v):
    return np.percentile(v, [2.5, 97.5])


def rodar(fase):
    os.makedirs(OUT, exist_ok=True)
    df = carregar()
    sp = splits(df)

    ajustes, grade, fixos = {}, [], []
    for s, cfgs in CONFIGS.items():
        for modelo, etapa in cfgs:
            gs, (t_tri, t_comp) = ajustar(df, sp[s][0], modelo, etapa)
            ajustes[(s, modelo, etapa)] = (gs, t_tri, t_comp)
            r = pd.DataFrame(gs.cv_results_)
            grade.append(r[["params", "mean_test_score", "std_test_score"]].assign(
                split=s, modelo=modelo, etapa=etapa))
            fixos.append({"split": s, "modelo": modelo, "etapa": etapa,
                          "auc_cv": gs.best_score_, "params": json.dumps(gs.best_params_),
                          "t_triagem": t_tri, "t_comp": t_comp})
            print(f"{s:15} {modelo:12} {etapa:6} AUC-CV {gs.best_score_:.3f}")
    fixos = pd.DataFrame(fixos)

    if fase == "cv":
        pd.concat(grade).to_csv(os.path.join(OUT, "cv_grade.csv"), index=False)
        fixos.to_csv(os.path.join(OUT, "cv_fixados.csv"), index=False)
    else:
        registrado = pd.read_csv(os.path.join(OUT, "cv_fixados.csv"))
        assert registrado["params"].tolist() == fixos["params"].tolist(), "hiperparametros mudaram"
        assert np.allclose(registrado[["t_triagem", "t_comp"]], fixos[["t_triagem", "t_comp"]]), \
            "limiares mudaram"
        avaliar_teste(df, sp, ajustes)
    with open(os.path.join(OUT, f"execucao_{fase}.json"), "w", encoding="utf-8") as fh:
        json.dump({"fase": fase, "data": datetime.now().isoformat(timespec="seconds"),
                   "commit": git("rev-parse", "HEAD"), "seed": SEED, "n_boot": N_BOOT,
                   "python": platform.python_version(), "sklearn": sklearn.__version__,
                   "pandas": pd.__version__}, fh, indent=2)


def avaliar_teste(df, sp, ajustes):
    rng = np.random.default_rng(SEED)
    boot = {s: rng.integers(0, len(te), (N_BOOT, len(te))) for s, (_, te) in sp.items()}
    linhas, auc_boot, prob = [], {}, {}
    for (s, modelo, etapa), (gs, t_tri, t_comp) in ajustes.items():
        te = sp[s][1]
        y = df.loc[te, ALVO].to_numpy()
        p = gs.predict_proba(preparar(df.loc[te], etapa))[:, 1]
        prob[(s, modelo, etapa)] = p
        pontos = metricas(y, p, t_tri, t_comp)
        amostras = pd.DataFrame([metricas(y[i], p[i], t_tri, t_comp) for i in boot[s]])
        auc_boot[(s, modelo, etapa)] = amostras["auc_roc"].to_numpy()
        for m, v in pontos.items():
            lo, hi = ic(amostras[m])
            linhas.append({"split": s, "modelo": modelo, "etapa": etapa, "metrica": m,
                           "valor": v, "ic_inf": lo, "ic_sup": hi})
    pd.DataFrame(linhas).to_csv(os.path.join(OUT, "resultados_teste.csv"), index=False)

    hip = []

    def delta(h, s, modelo, a, b, va, vb):
        d = va - vb
        lo, hi = ic(d)
        hip.append({"hipotese": h, "split": s, "modelo": modelo, "comparacao": f"{a} - {b}",
                    "delta_auc": d.mean(), "ic_inf": lo, "ic_sup": hi, "exclui_zero": lo > 0 or hi < 0})

    for s in ["aleatorio", "temporal"]:
        for m in ["RL", "GB"]:
            A = lambda e: auc_boot[(s, m, e)]  # noqa: E731
            delta("H1", s, m, "E0", "idade", A("E0"), auc_boot[(s, "idade", "idade")])
            delta("H1", s, m, "E1", "E0", A("E1"), A("E0"))
            delta("verificacao", s, m, "E2", "E1", A("E2"), A("E1"))
            delta("ablacao", s, m, "E0", "E0-", A("E0"), A("E0-"))
            delta("ablacao", s, m, "E1", "E1-", A("E1"), A("E1-"))
    te = sp["aleatorio"][1]
    y, idade = df.loc[te, ALVO].to_numpy(), df.loc[te, "idade"].to_numpy()
    for m in ["RL", "GB"]:
        p = prob[("aleatorio", m, "E1")]
        auc_faixa = {nome: np.array([roc_auc_score(y[i][mask[i]], p[i][mask[i]])
                                     for i in boot["aleatorio"]])
                     for nome, mask in [("<2", idade < 2), ("2-19", idade >= 2)]}
        delta("H3", "aleatorio", m, "E1 <2", "E1 2-19", auc_faixa["<2"], auc_faixa["2-19"])
        delta("H4", "-", m, "E1 temporal", "E1 aleatorio",
              auc_boot[("temporal", m, "E1")], auc_boot[("aleatorio", m, "E1")])
    pd.DataFrame(hip).to_csv(os.path.join(OUT, "hipoteses.csv"), index=False)

    # interpretacao: melhor modelo E1 (pela AUC-CV) no split aleatorio
    m = max(["RL", "GB"], key=lambda k: ajustes[("aleatorio", k, "E1")][0].best_score_)
    gs, t_tri, _ = ajustes[("aleatorio", m, "E1")]
    pos = prob[("aleatorio", m, "E1")] >= t_tri
    pd.crosstab(pd.Series(y, name="real"), pd.Series(pos.astype(int), name="previsto")
                ).to_csv(os.path.join(OUT, f"matriz_confusao_{m}_E1.csv"))
    X = preparar(df.loc[te], "E1")
    imp = permutation_importance(gs.best_estimator_, X, y, scoring="roc_auc", n_repeats=20,
                                 random_state=SEED, n_jobs=-1)
    pd.DataFrame({"atributo": X.columns, "queda_auc": imp.importances_mean,
                  "desvio": imp.importances_std}).sort_values("queda_auc", ascending=False
                  ).to_csv(os.path.join(OUT, f"importancia_{m}_E1.csv"), index=False)


def git(*args):
    return subprocess.run(["git", *args], cwd=BASE, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", choices=["cv", "teste"], required=True)
    fase = ap.parse_args().fase
    if fase == "teste" and git("status", "--porcelain", "--", SCRIPT, "output/modelagem/cv_fixados.csv"):
        sys.exit(f"{SCRIPT} ou cv_fixados.csv com alteracoes nao commitadas: "
                 "faca o commit do pre-registro antes de tocar o teste.")
    rodar(fase)
