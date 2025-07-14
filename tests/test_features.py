import pandas as pd
import numpy as np
from src.features import adicionar_features_basicas


def gerar_dataframe_exemplo() -> pd.DataFrame:
    """
    Gera um DataFrame de exemplo com colunas mínimas para o teste da função adicionar_features_basicas.
    """
    np.random.seed(42)
    datas = pd.date_range(start="2022-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {
            "Data": datas,
            "open": np.random.rand(40) * 100,
            "high": np.random.rand(40) * 110,
            "low": np.random.rand(40) * 90,
            "close": np.random.rand(40) * 105,
            "volume": np.random.randint(1000, 10000, size=40),
        }
    )
    return df


def test_adicionar_features_basicas() -> None:
    """
    Testa se a função adicionar_features_basicas gera corretamente as novas colunas esperadas
    a partir de um DataFrame de exemplo com dados simulados.
    """
    df_exemplo = gerar_dataframe_exemplo()
    df_resultado = adicionar_features_basicas(df_exemplo)

    # Colunas esperadas que devem ter sido adicionadas
    colunas_esperadas = [
        "Retorno",
        "TendenciaAlta",
        "Amplitude",
        "Variacao_Alta_Abertura",
        "Variacao_Abertura_Baixa",
        "Razao_Alta_Baixa",
        "Razao_Retorno_Desvio",
        "MediaMovel_7d",
        "MediaMovel_14d",
        "MediaMovel_30d",
        "DesvioPadrao_7d",
        "DesvioPadrao_14d",
        "MediaVolume_7d",
        "MediaVolume_14d",
        "Retorno_Anterior_1d",
        "Retorno_Anterior_2d",
        "DiaSemana",
        "Mes",
        "FimDeSemana",
    ]

    for coluna in colunas_esperadas:
        assert (
            coluna in df_resultado.columns
        ), f"Coluna esperada '{coluna}' não encontrada no resultado."

    # Verifica se o DataFrame final não está vazio
    assert not df_resultado.empty, "DataFrame resultante está vazio."

    # Verifica se não há NaNs restantes
    assert (
        df_resultado.isna().sum().sum() == 0
    ), "Ainda existem valores NaN no DataFrame final."
