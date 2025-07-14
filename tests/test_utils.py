import os
import pandas as pd
import numpy as np
import joblib
import pytest
from sklearn.linear_model import LinearRegression
from pandas import DataFrame
from typing import Generator

from src.utils import (
    plot_grafico_retorno,
    salvar_modelo,
    carregar_modelo,
    preprocessar_dados,
    salvar_medidas_dispersao,
)


@pytest.fixture
def df_resultados_exemplo() -> Generator[DataFrame, None, None]:
    """
    Fixture que retorna um DataFrame de exemplo com retornos simulados.
    """
    df = pd.DataFrame(
        {
            "Criptomoeda": ["BTC", "ETH"],
            "RetornoPercentual_MLP": [12.5, -3.2],
            "RetornoPercentual_Linear": [10.1, 2.0],
        }
    )
    yield df


def test_plot_grafico_retorno(df_resultados_exemplo: DataFrame) -> None:
    """
    Testa a geração e salvamento do gráfico de retorno percentual.
    """
    modelo = "MLP"
    plot_grafico_retorno(df_resultados_exemplo, modelo=modelo)
    caminho = f"figures/retornos_criptos_{modelo}.png"
    assert os.path.exists(caminho)
    os.remove(caminho)


def test_salvar_e_carregar_modelo(tmp_path: str) -> None:
    """
    Testa se um modelo LinearRegression é salvo e carregado corretamente.
    """
    modelo = LinearRegression()
    nome = "modelo_teste"
    pasta = tmp_path / "modelos"

    salvar_modelo(modelo, nome, pasta=str(pasta))
    caminho = pasta / f"{nome}.joblib"
    assert caminho.exists()

    modelo_carregado = carregar_modelo(nome, pasta=str(pasta))
    assert isinstance(modelo_carregado, LinearRegression)


def test_preprocessar_dados() -> None:
    """
    Testa o pré-processamento de dados em um DataFrame simulado.
    """
    datas = pd.date_range("2023-01-01", periods=40)
    df = pd.DataFrame(
        {
            "Data": datas,
            "open": np.random.rand(40) * 100,
            "high": np.random.rand(40) * 100,
            "low": np.random.rand(40) * 100,
            "close": np.random.rand(40) * 100,
            "Volume": np.random.randint(1000, 5000, size=40),
        }
    )

    df_processado = preprocessar_dados(df)
    assert isinstance(df_processado, pd.DataFrame)
    assert "Fechamento" in df_processado.columns
    assert df_processado.isna().sum().sum() == 0
