import numpy as np
import pandas as pd
from src.evaluation import simular_estrategia_investimento

def test_simulacao_lucro_calculado_corretamente():
    
    df = pd.DataFrame({
        "Data": pd.date_range(start="2023-01-01", periods=5),
        "Fechamento": [100, 102, 105, 103, 106]
    })
    
    previsoes = np.array([101, 103, 106, 105, 107])

    lucro, resultado = simular_estrategia_investimento(df, previsoes, threshold=0.01, verbose=False)

    assert isinstance(lucro, float)
    assert isinstance(resultado, pd.DataFrame)
    assert not resultado.empty
    assert "CapitalFinal" in resultado.columns
    assert "RetornoPercentual" in resultado.columns

def test_simulacao_sem_compras_quando_previsao_baixa():
    df = pd.DataFrame({
        "Data": pd.date_range(start="2023-01-01", periods=5),
        "Fechamento": [100, 102, 104, 106, 108]
    })

    previsoes = np.array([100, 101, 102, 104, 107])

    lucro, resultado = simular_estrategia_investimento(df, previsoes, threshold=0.05, verbose=False)

    assert lucro == 0
    assert resultado.empty or "Lucro" not in resultado.columns or resultado["Lucro"].sum() == 0
