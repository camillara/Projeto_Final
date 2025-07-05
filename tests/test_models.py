import pandas as pd
from src.models import treinar_modelos

def test_treinar_modelos():
    df = pd.DataFrame({
        "Data": pd.date_range("2023-01-01", periods=20, freq="D"),
        "Fechamento": range(20),
        "Retorno": range(20),
        "MediaMovel_7d": range(20),
        "DesvioPadrao_7d": range(20),
        "TendenciaAlta": [True, False] * 10
    })
    resultados = treinar_modelos(df, nome_cripto="teste", reutilizar=False)
    assert "MLP" in resultados
    assert "Linear" in resultados
    assert "Polinomial_2" in resultados
    assert resultados["MLP"]["mse"] is not None
