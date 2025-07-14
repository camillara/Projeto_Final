from typing import Optional, List, Dict, Any
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from src.utils import preprocessar_dados, carregar_modelo, salvar_modelo
from sklearn.preprocessing import MinMaxScaler
from src.visualization import (
    salvar_graficos_mlp,
    salvar_importancia_features,
    salvar_graficos_regressao,
)
from sklearn.feature_selection import SelectKBest, f_regression


def treinar_modelos(
    df: pd.DataFrame,
    target_col: str = "Fechamento",
    num_folds: int = 5,
    nome_cripto: str = "modelo",
    reutilizar: bool = True,
    modelos_especificos: Optional[List[str]] = None,
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
        mse_lr = -cross_val_score(
            modelo_lr, X, y, cv=kf, scoring="neg_mean_squared_error"
        ).mean()
        preds_lr = modelo_lr.predict(X)
        resultados["Linear"] = {
            "modelo": modelo_lr,
            "mse": mse_lr,
            "y_real": y.values,
            "previsoes": preds_lr,
        }

        salvar_graficos_regressao(
            nome_modelo="Linear",
            y_real=y.values,
            y_pred=preds_lr,
            nome_cripto=nome_cripto,
        )

        logging.info(f"[MODELOS] Regressão Linear: MSE médio = {mse_lr:.4f}")

    # Polynomial Regressions
    k_melhores = 3
    seletor = SelectKBest(score_func=f_regression, k=k_melhores)
    X_reduzido = seletor.fit_transform(X, y)
    nomes_features_selecionadas = X.columns[seletor.get_support()].tolist()

    logging.info(
        f"[FEATURES] Selecionadas para Polynomial Regression: {nomes_features_selecionadas}"
    )

    for grau in range(2, 11):
        modelo_nome = f"POLINOMIAL_{grau}"
        if modelos_especificos and modelo_nome not in modelos_especificos:
            continue

        nome_modelo_poly = f"{nome_cripto}_polinomial_grau{grau}"
        modelo_poly = carregar_modelo(nome_modelo_poly) if reutilizar else None

        if modelo_poly is None:
            modelo_poly = make_pipeline(
                PolynomialFeatures(degree=grau), LinearRegression()
            )
            modelo_poly.fit(X_reduzido, y)
            salvar_modelo(modelo_poly, nome_modelo_poly)

        mse_poly = -cross_val_score(
            modelo_poly, X_reduzido, y, cv=kf, scoring="neg_mean_squared_error"
        ).mean()
        preds_poly = modelo_poly.predict(X_reduzido)

        resultados[modelo_nome] = {
            "modelo": modelo_poly,
            "mse": mse_poly,
            "y_real": y.values,
            "previsoes": preds_poly,
        }

        salvar_graficos_regressao(
            nome_modelo=modelo_nome,
            y_real=y.values,
            y_pred=preds_poly,
            nome_cripto=nome_cripto,
        )

        logging.info(f"[MODELOS] Polinomial Grau {grau}: MSE médio = {mse_poly:.4f}")

    # MLP Regressor
    if not modelos_especificos or "MLP" in modelos_especificos:
        nome_modelo_mlp = f"{nome_cripto}_mlp"
        modelo_mlp = carregar_modelo(nome_modelo_mlp) if reutilizar else None

        # Escalonar os dados
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        if modelo_mlp is None:
            modelo_mlp = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation="relu",
                solver="adam",
                learning_rate="adaptive",
                learning_rate_init=0.0005,
                alpha=0.0005,
                max_iter=3000,
                early_stopping=True,
                n_iter_no_change=600,
                validation_fraction=0.2,
                shuffle=True,
                random_state=42,
            )

            modelo_mlp.fit(X_scaled, y)
            salvar_modelo(modelo_mlp, nome_modelo_mlp)

        mse_mlp = -cross_val_score(
            modelo_mlp, X_scaled, y, cv=kf, scoring="neg_mean_squared_error"
        ).mean()
        preds_mlp = modelo_mlp.predict(X_scaled)

        resultados["MLP"] = {
            "modelo": modelo_mlp,
            "mse": mse_mlp,
            "y_real": y.values,
            "previsoes": preds_mlp,
        }
        logging.info(f"[MODELOS] MLP Regressor: MSE médio = {mse_mlp:.4f}")

        salvar_graficos_mlp(
            y_real=y.values,
            y_pred=preds_mlp,
            loss_curve=modelo_mlp.loss_curve_,
            nome_cripto=nome_cripto,
        )

        salvar_importancia_features(
            modelo=modelo_mlp,
            X_scaled=X_scaled,
            y=y.values,
            feature_names=X.columns.tolist(),
            nome_cripto=nome_cripto,
        )

    logging.info(
        f"[FEATURES] Features utilizadas para treinamento: {X.columns.tolist()}"
    )
    logging.info("[MODELOS] Treinamento concluído com sucesso.")
    return resultados
