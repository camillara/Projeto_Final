import numpy as np
import pandas as pd
from src.evaluation import simular_estrategia_investimento

def test_simulacao_lucro_calculado_corretamente() -> None:
    """
    Testa a função `simular_estrategia_investimento` com previsões que devem gerar lucro.
    
    Verifica:
    - Se o lucro é um float.
    - Se o DataFrame de resultados não está vazio.
    - Se contém as colunas 'CapitalFinal' e 'RetornoPercentual'.
    """
    df = pd.DataFrame({
        "Data": pd.date_range(start="2023-01-01", periods=5),
        "Fechamento": [100, 102, 105, 103, 106]
    })
    
    previsoes = np.array([101, 103, 106, 105, 107])

    lucro, resultado = simular_estrategia_investimento(df, previsoes, threshold=0.01, verbose=False)

    assert isinstance(lucro, float), "Lucro deve ser do tipo float"
    assert isinstance(resultado, pd.DataFrame), "Resultado deve ser um DataFrame"
    assert not resultado.empty, "Resultado não pode estar vazio"
    assert "CapitalFinal" in resultado.columns, "Coluna 'CapitalFinal' ausente"
    assert "RetornoPercentual" in resultado.columns, "Coluna 'RetornoPercentual' ausente"

def test_simulacao_sem_compras_quando_previsao_baixa() -> None:
    """
    Testa `simular_estrategia_investimento` com previsões que não ultrapassam o threshold,
    garantindo que nenhuma compra seja realizada.

    Verifica:
    - Se o lucro é zero.
    - Se o DataFrame está vazio ou sem lucros registrados.
    """
    df = pd.DataFrame({
        "Data": pd.date_range(start="2023-01-01", periods=5),
        "Fechamento": [100, 102, 104, 106, 108]
    })

    previsoes = np.array([100, 101, 102, 104, 107])

    lucro, resultado = simular_estrategia_investimento(df, previsoes, threshold=0.05, verbose=False)

    assert lucro == 0, "Lucro deveria ser zero quando nenhuma compra é realizada"
    assert resultado.empty or "Lucro" not in resultado.columns or resultado["Lucro"].sum() == 0, \
        "Resultado não deveria conter lucro quando não há compras"
