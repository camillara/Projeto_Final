import pandas as pd
from src.features import adicionar_features_basicas

def test_adicionar_features_basicas():
    df = pd.DataFrame({
        "Data": pd.date_range("2023-01-01", periods=10, freq="D"),
        "Fechamento": range(10)
    })
    df_novo = adicionar_features_basicas(df)
    assert "Retorno" in df_novo.columns
    assert "MediaMovel_7d" in df_novo.columns
    assert "DesvioPadrao_7d" in df_novo.columns
    assert "TendenciaAlta" in df_novo.columns
    assert not df_novo.empty
