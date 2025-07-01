import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline

from src.utils import salvar_modelo, carregar_modelo

# Configuração do logger
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def treinar_modelos(
    df: pd.DataFrame,
    target_col: str = "Fechamento",
    k_folds: int = 5,
    nome_cripto: str = "modelo",
    reutilizar: bool = True
) -> Dict[str, Any]:
    """
    Treina e avalia modelos ou reutiliza versões salvos.

    Args:
        df (pd.DataFrame): Dados com features.
        target_col (str): Coluna alvo (padrão: 'Fechamento').
        k_folds (int): Número de folds para cross-validation.
        nome_cripto (str): Nome base para salvar/carregar modelos.
        reutilizar (bool): Se True, tenta carregar modelos salvos.

    Returns:
        Dict[str, Any]: Dicionário com modelos e MSEs.
    """
    resultados: Dict[str, Any] = {}

    X = df.drop(columns=[target_col, "Data"], errors="ignore")
    y = df[target_col]
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    logging.info("[MODELOS] Iniciando treinamento com K-Fold cross-validation...")

    # Regressão Linear
    nome_modelo_lr = f"{nome_cripto}_linear"
    modelo_lr = carregar_modelo(nome_modelo_lr) if reutilizar else None
    if modelo_lr is None:
        modelo_lr = LinearRegression()
        modelo_lr.fit(X, y)
        salvar_modelo(modelo_lr, nome_modelo_lr)
    mse_lr = -cross_val_score(modelo_lr, X, y, cv=kf, scoring='neg_mean_squared_error').mean()
    resultados["Linear"] = {"modelo": modelo_lr, "mse": mse_lr}
    logging.info(f"[MODELOS] Regressão Linear: MSE médio = {mse_lr:.4f}")

    # Regressões Polinomiais
    for grau in range(2, 11):
        nome_modelo_poly = f"{nome_cripto}_polinomial_grau{grau}"
        modelo_poly = carregar_modelo(nome_modelo_poly) if reutilizar else None
        if modelo_poly is None:
            modelo_poly = make_pipeline(
                PolynomialFeatures(degree=grau),
                LinearRegression()
            )
            modelo_poly.fit(X, y)
            salvar_modelo(modelo_poly, nome_modelo_poly)
        mse_poly = -cross_val_score(modelo_poly, X, y, cv=kf, scoring='neg_mean_squared_error').mean()
        resultados[f"Polinomial_{grau}"] = {"modelo": modelo_poly, "mse": mse_poly}
        logging.info(f"[MODELOS] Polinomial Grau {grau}: MSE médio = {mse_poly:.4f}")

    # MLP
    nome_modelo_mlp = f"{nome_cripto}_mlp"
    modelo_mlp = carregar_modelo(nome_modelo_mlp) if reutilizar else None
    if modelo_mlp is None:
        modelo_mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        modelo_mlp.fit(X, y)
        salvar_modelo(modelo_mlp, nome_modelo_mlp)
    mse_mlp = -cross_val_score(modelo_mlp, X, y, cv=kf, scoring='neg_mean_squared_error').mean()
    resultados["MLP"] = {"modelo": modelo_mlp, "mse": mse_mlp}
    logging.info(f"[MODELOS] MLP Regressor: MSE médio = {mse_mlp:.4f}")

    logging.info("[MODELOS] Treinamento concluído com sucesso.")
    return resultados