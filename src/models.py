import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline


def treinar_modelos(
    df: pd.DataFrame,
    target_col: str = "Fechamento",
    k_folds: int = 5
) -> Dict[str, Any]:
    """
    Treina MLPRegressor, Regressão Linear e Polinomial (graus 2 a 10).
    Aplica K-Fold Cross Validation para comparar desempenho.

    Args:
        df (pd.DataFrame): DataFrame com features e target.
        target_col (str): Nome da coluna alvo (default: 'Fechamento').
        k_folds (int): Número de folds na validação cruzada.

    Returns:
        Dict[str, Any]: Resultados com modelos, scores e pipelines.
    """
    resultados = {}

    X = df.drop(columns=[target_col, "Data"], errors="ignore")
    y = df[target_col]

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Regressão Linear
    modelo_lr = LinearRegression()
    scores_lr = cross_val_score(modelo_lr, X, y, cv=kf, scoring='neg_mean_squared_error')
    resultados["Linear"] = {
        "modelo": modelo_lr,
        "mse": -scores_lr.mean()
    }

    # Regressão Polinomial (graus 2 a 10)
    for grau in range(2, 11):
        poly_model = make_pipeline(
            PolynomialFeatures(degree=grau),
            LinearRegression()
        )
        scores_poly = cross_val_score(poly_model, X, y, cv=kf, scoring='neg_mean_squared_error')
        resultados[f"Polinomial_{grau}"] = {
            "modelo": poly_model,
            "mse": -scores_poly.mean()
        }

    # MLP Regressor
    modelo_mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    scores_mlp = cross_val_score(modelo_mlp, X, y, cv=kf, scoring='neg_mean_squared_error')
    resultados["MLP"] = {
        "modelo": modelo_mlp,
        "mse": -scores_mlp.mean()
    }

    return resultados
