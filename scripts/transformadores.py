# -*- coding: utf-8 -*-
"""
Transformadores scikit-learn proprios do projeto.

Ficam em modulo separado (e nao dentro de 04_transformacoes.py) para que o
preprocessor.joblib possa ser carregado por qualquer script que tenha
scripts/ no sys.path — o pickle guarda a referencia "transformadores.X".
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ImputadorPorFaixa(BaseEstimator, TransformerMixin):
    """
    Imputa colunas numericas pela mediana da faixa etaria, aprendida no treino.
    Sem faixa (idade ausente), usa a mediana global do treino. Recebe as
    numericas + a coluna de faixa e devolve so as numericas.
    """

    def __init__(self, grupo="faixa_etaria"):
        self.grupo = grupo

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.colunas_ = [c for c in X.columns if c != self.grupo]
        self.medianas_ = X.groupby(self.grupo)[self.colunas_].median()
        self.global_ = X[self.colunas_].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.colunas_:
            por_faixa = X[self.grupo].map(self.medianas_[c])
            X[c] = X[c].fillna(por_faixa).fillna(self.global_[c])
        return X[self.colunas_].to_numpy(float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.colunas_, dtype=object)
