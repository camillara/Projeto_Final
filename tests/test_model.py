import pandas as pd
import numpy as np
from typing import Any, Dict
from unittest.mock import patch
from src.models import treinar_modelos

def gerar_dataframe_teste() -> pd.DataFrame:
    """
    Gera um DataFrame fictício para testes com colunas esperadas.
    """
    np.random.seed(42)
    datas = pd.date_range(start="2022-01-01", periods=100, freq='D')
    df = pd.DataFrame({
        "Data": datas,
        "Fechamento": np.random.rand(100) * 1000,
        "Volume": np.random.rand(100) * 10000,
        "Retorno": np.random.rand(100),
        "MediaMovel_7d": np.random.rand(100),
        "DesvioPadrao_7d": np.random.rand(100),
        "TendenciaAlta": np.random.randint(0, 2, 100)
    })
    return df

@patch("src.models.carregar_modelo", return_value=None)
@patch("src.models.salvar_modelo")
@patch("src.models.salvar_graficos_regressao")
@patch("src.models.salvar_graficos_mlp")
@patch("src.models.salvar_importancia_features")
@patch("src.models.preprocessar_dados", side_effect=lambda df: df)
def test_treinar_modelos_linear_mlp_poly(
    mock_preprocessar,
    mock_importancia,
    mock_mlp_grafico,
    mock_regressao,
    mock_salvar_modelo,
    mock_carregar_modelo
) -> None:
    """
    Testa o treinamento dos modelos Linear, MLP e Polinomial Grau 2 com um DataFrame de exemplo.
    Garante que os resultados retornados contenham os modelos treinados e métricas esperadas.
    """
    df_teste = gerar_dataframe_teste()

    resultados: Dict[str, Any] = treinar_modelos(
        df_teste,
        target_col="Fechamento",
        num_folds=3,
        nome_cripto="TESTE",
        reutilizar=False,
        modelos_especificos=["LINEAR", "MLP", "POLINOMIAL_2"]
    )

    assert "Linear" in resultados
    assert "MLP" in resultados
    assert "POLINOMIAL_2" in resultados

    for modelo in ["Linear", "MLP", "POLINOMIAL_2"]:
        assert "modelo" in resultados[modelo]
        assert "mse" in resultados[modelo]
        assert "y_real" in resultados[modelo]
        assert "previsoes" in resultados[modelo]
        assert isinstance(resultados[modelo]["mse"], float)
        assert len(resultados[modelo]["y_real"]) == len(resultados[modelo]["previsoes"])
