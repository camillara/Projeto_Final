import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error


def simular_estrategia_investimento(
    df: pd.DataFrame,
    previsoes: np.ndarray,
    threshold: float = 0.01,
    preco_col: str = "Fechamento",
    verbose: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """
    Simula uma estratégia de investimento com base nas previsões de preço.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados reais.
        previsoes (np.ndarray): Previsões do modelo.
        threshold (float): Variação mínima prevista para considerar investimento.
        preco_col (str): Coluna de preço a ser usada como referência.
        verbose (bool): Indica se deve imprimir logs da simulação.

    Returns:
        Tuple[float, pd.DataFrame]: Lucro total e DataFrame com resultados da simulação.
    """
    capital = 10000
    capital_disponivel = capital
    lucro_total = 0
    historico = []

    for i in range(len(previsoes) - 1):
        preco_hoje = df[preco_col].iloc[i]
        preco_amanha = df[preco_col].iloc[i + 1]
        previsao = previsoes[i]

        variacao_prevista = (previsao - preco_hoje) / preco_hoje

        if variacao_prevista > threshold:
            quantidade_comprada = capital_disponivel / preco_hoje
            capital_final = quantidade_comprada * preco_amanha
            lucro = capital_final - capital_disponivel

            lucro_total += lucro
            historico.append(
                {
                    "Data": df["Data"].iloc[i],
                    "PrecoHoje": preco_hoje,
                    "PrecoAmanha": preco_amanha,
                    "Previsao": previsao,
                    "Lucro": round(lucro, 2),
                }
            )

    df_resultado = pd.DataFrame(historico)

    if not df_resultado.empty:
        df_resultado["CapitalFinal"] = df_resultado["Lucro"].cumsum() + capital
        df_resultado["RetornoPercentual"] = (
            (df_resultado["CapitalFinal"] - capital) / capital * 100
        )
    else:
        df_resultado = pd.DataFrame(
            {"CapitalFinal": [capital], "RetornoPercentual": [0.0]}
        )

    if verbose:
        logging.info(
            f"[SIMULAÇÃO] Lucro final com capital inicial de R$ {capital:.2f}: R$ {lucro_total:.2f}"
        )
        logging.info(
            f"[SIMULAÇÃO] Retorno percentual: {lucro_total / capital * 100:.2f}%"
        )

    return lucro_total, df_resultado


def comparar_modelos_regressao(
    df: pd.DataFrame, y_real: np.ndarray, mlp_preds: np.ndarray
) -> Dict[str, Dict]:
    """
    Compara diferentes modelos de regressão (linear, polinomiais e MLP) com base em métricas e simulação de lucro.

    Args:
        df (pd.DataFrame): DataFrame original com os dados do preço.
        y_real (np.ndarray): Valores reais observados (ex: preços reais).
        mlp_preds (np.ndarray): Previsões do modelo MLP.

    Returns:
        Dict[str, Dict]: Dicionário com métricas, previsões e resultados de simulação de cada modelo.
    """
    resultados = {}
    X = np.arange(len(y_real)).reshape(-1, 1)

    # === Regressão Linear ===
    modelo_linear = LinearRegression()
    modelo_linear.fit(X, y_real)
    pred_linear = modelo_linear.predict(X)
    lucro_linear, sim_linear = simular_estrategia_investimento(
        df, pred_linear, verbose=False
    )
    resultados["Linear"] = {
        "modelo": modelo_linear,
        "previsoes": pred_linear,
        "lucro": lucro_linear,
        "simulacao": sim_linear,
        "mse": mean_squared_error(y_real, pred_linear),
        "mae": mean_absolute_error(y_real, pred_linear),
        "correlacao": np.corrcoef(y_real, pred_linear)[0, 1],
    }

    # === Regressões Polinomiais (grau 2 a 10) ===
    for grau in range(2, 11):
        modelo_poly = make_pipeline(PolynomialFeatures(grau), LinearRegression())
        modelo_poly.fit(X, y_real)
        pred_poly = modelo_poly.predict(X)
        lucro_poly, sim_poly = simular_estrategia_investimento(
            df, pred_poly, verbose=False
        )
        resultados[f"Poly_{grau}"] = {
            "modelo": modelo_poly,
            "previsoes": pred_poly,
            "lucro": lucro_poly,
            "simulacao": sim_poly,
            "mse": mean_squared_error(y_real, pred_poly),
            "mae": mean_absolute_error(y_real, pred_poly),
            "correlacao": np.corrcoef(y_real, pred_poly)[0, 1],
        }

    # === Modelo MLP (já fornecido) ===
    lucro_mlp, sim_mlp = simular_estrategia_investimento(df, mlp_preds, verbose=False)
    resultados["MLP"] = {
        "previsoes": mlp_preds,
        "lucro": lucro_mlp,
        "simulacao": sim_mlp,
        "mse": mean_squared_error(y_real, mlp_preds),
        "mae": mean_absolute_error(y_real, mlp_preds),
        "correlacao": np.corrcoef(y_real, mlp_preds)[0, 1],
    }

    return resultados
