import numpy as np
import pandas as pd
from src.evaluation import comparar_modelos_regressao

def test_comparar_modelos_regressao() -> None:
    """
    Testa a função `comparar_modelos_regressao` com dados simulados.

    Verifica:
    - Se todos os modelos esperados estão presentes no dicionário de resultados.
    - Se os campos obrigatórios estão presentes para cada modelo.
    - Se as métricas (mse, mae, correlacao) são valores float.
    """
    # Simulação de dados de entrada
    df = pd.DataFrame({
        "Data": pd.date_range(start="2023-01-01", periods=20),
        "Fechamento": np.linspace(100, 120, 20)
    })

    y_real = df["Fechamento"].values
    mlp_preds = y_real + np.random.normal(0, 0.5, size=len(y_real))

    resultados = comparar_modelos_regressao(df, y_real, mlp_preds)

    # Modelos esperados
    modelos_esperados = ["Linear"] + [f"Poly_{i}" for i in range(2, 11)] + ["MLP"]
    for modelo in modelos_esperados:
        assert modelo in resultados, f"Modelo '{modelo}' ausente nos resultados"

        resultado = resultados[modelo]
        assert "previsoes" in resultado, f"Campo 'previsoes' ausente para {modelo}"
        assert "lucro" in resultado, f"Campo 'lucro' ausente para {modelo}"
        assert "simulacao" in resultado, f"Campo 'simulacao' ausente para {modelo}"
        assert "mse" in resultado, f"Campo 'mse' ausente para {modelo}"
        assert "mae" in resultado, f"Campo 'mae' ausente para {modelo}"
        assert "correlacao" in resultado, f"Campo 'correlacao' ausente para {modelo}"

        assert isinstance(resultado["mse"], float), f"'mse' deve ser float em {modelo}"
        assert isinstance(resultado["mae"], float), f"'mae' deve ser float em {modelo}"
        assert isinstance(resultado["correlacao"], float), f"'correlacao' deve ser float em {modelo}"
        assert isinstance(resultado["simulacao"], pd.DataFrame), f"'simulacao' deve ser DataFrame em {modelo}"
