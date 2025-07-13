from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from src.utils import preprocessar_dados, carregar_modelo, salvar_modelo

def treinar_modelos(
    df: pd.DataFrame,
    target_col: str = "Fechamento",
    num_folds: int = 5,
    nome_cripto: str = "modelo",
    reutilizar: bool = True,
    modelos_especificos: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Treina e avalia modelos de regressão (Linear, Polinomial, MLP) com K-Fold CV ou reutiliza versões salvos.

    Args:
        df (pd.DataFrame): Dados com features.
        target_col (str): Coluna alvo (padrão: 'Fechamento').
        num_folds (int): Número de folds para cross-validation.
        nome_cripto (str): Nome base para salvar/carregar modelos.
        reutilizar (bool): Se True, tenta carregar modelos salvos.
        modelos_especificos (List[str] | None): Lista com nomes dos modelos a treinar. 
            Ex: ["LINEAR", "POLINOMIAL_2", "MLP"]. Se None, todos são treinados.

    Returns:
        Dict[str, Any]: Dicionário com modelos, MSEs e previsões.
    """
    resultados: Dict[str, Any] = {}

    modelos_validos = ["LINEAR", "MLP"] + [f"POLINOMIAL_{i}" for i in range(2, 11)]
    if modelos_especificos:
        modelos_upper = [m.upper() for m in modelos_especificos]
        for modelo in modelos_upper:
            if modelo not in modelos_validos:
                raise ValueError(f"[ERRO] Modelo '{modelo}' não é suportado.")
        modelos_especificos = modelos_upper

    try:
        df = preprocessar_dados(df)
    except Exception as e:
        logging.error(f"[ERRO] Falha no preprocessamento para {nome_cripto}: {e}")
        return resultados

    X = df.drop(columns=[target_col, "Data"], errors="ignore")
    y = df[target_col]
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    logging.info("[MODELOS] Iniciando treinamento com K-Fold cross-validation...")

    # Linear Regression
    if not modelos_especificos or "LINEAR" in modelos_especificos:
        nome_modelo_lr = f"{nome_cripto}_linear"
        modelo_lr = carregar_modelo(nome_modelo_lr) if reutilizar else None
        if modelo_lr is None:
            modelo_lr = LinearRegression()
            modelo_lr.fit(X, y)
            salvar_modelo(modelo_lr, nome_modelo_lr)
        mse_lr = -cross_val_score(modelo_lr, X, y, cv=kf, scoring='neg_mean_squared_error').mean()
        preds_lr = modelo_lr.predict(X)
        resultados["Linear"] = {
            "modelo": modelo_lr,
            "mse": mse_lr,
            "y_real": y.values,
            "previsoes": preds_lr
        }
        logging.info(f"[MODELOS] Regressão Linear: MSE médio = {mse_lr:.4f}")

    # Polynomial Regressions
    for grau in range(2, 11):
        modelo_nome = f"POLINOMIAL_{grau}"
        if modelos_especificos and modelo_nome not in modelos_especificos:
            continue
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
        preds_poly = modelo_poly.predict(X)
        resultados[modelo_nome] = {
            "modelo": modelo_poly,
            "mse": mse_poly,
            "y_real": y.values,
            "previsoes": preds_poly
        }
        logging.info(f"[MODELOS] Polinomial Grau {grau}: MSE médio = {mse_poly:.4f}")

    # MLP Regressor
    if not modelos_especificos or "MLP" in modelos_especificos:
        nome_modelo_mlp = f"{nome_cripto}_mlp"
        modelo_mlp = carregar_modelo(nome_modelo_mlp) if reutilizar else None
        if modelo_mlp is None:
            modelo_mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
            modelo_mlp.fit(X, y)
            salvar_modelo(modelo_mlp, nome_modelo_mlp)
        mse_mlp = -cross_val_score(modelo_mlp, X, y, cv=kf, scoring='neg_mean_squared_error').mean()
        preds_mlp = modelo_mlp.predict(X)
        resultados["MLP"] = {
            "modelo": modelo_mlp,
            "mse": mse_mlp,
            "y_real": y.values,
            "previsoes": preds_mlp
        }
        logging.info(f"[MODELOS] MLP Regressor: MSE médio = {mse_mlp:.4f}")

    logging.info("[MODELOS] Treinamento concluído com sucesso.")
    return resultados
