import pandas as pd
import pytest
from src.features import adicionar_features_basicas

def test_adicionar_features_basicas() -> None:
    """
    Testa a função `adicionar_features_basicas` para garantir que:
    - As colunas de features esperadas sejam corretamente adicionadas.
    - O DataFrame de saída não esteja vazio.
    - As colunas adicionadas contenham dados (não sejam todas NaN).
    """
    df = pd.DataFrame({
        "Data": pd.date_range("2023-01-01", periods=10, freq="D"),
        "Fechamento": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    })

    df_novo = adicionar_features_basicas(df)

    # Colunas esperadas
    colunas_esperadas = ["Retorno", "MediaMovel_7d", "DesvioPadrao_7d", "TendenciaAlta"]
    
    for coluna in colunas_esperadas:
        assert coluna in df_novo.columns, f"Coluna esperada '{coluna}' não encontrada."

    # Verifica que o DataFrame resultante não está vazio
    assert not df_novo.empty, "O DataFrame resultante está vazio."

    # Verifica se as colunas adicionadas contêm dados (ou seja, não são todos NaN)
    for coluna in colunas_esperadas:
        assert df_novo[coluna].notna().any(), f"A coluna '{coluna}' contém apenas valores NaN."
        

def test_adicionar_features_basicas_sem_coluna_fechamento() -> None:
    """
    Testa se a função `adicionar_features_basicas` lança erro apropriado
    quando a coluna 'Fechamento' está ausente no DataFrame de entrada.
    """
    df = pd.DataFrame({
        "Data": pd.date_range("2023-01-01", periods=10, freq="D"),
        "Preço": range(10)  # coluna errada propositalmente
    })

    with pytest.raises(KeyError, match="Fechamento"):
        adicionar_features_basicas(df)