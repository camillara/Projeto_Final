import pandas as pd
import numpy as np 
from typing import Any, Dict

from src.models import treinar_modelos


def test_treinar_modelos() -> None:
    """
    Testa a função treinar_modelos() com um DataFrame sintético simples.

    Verifica se os principais modelos (MLP, Linear, Polinomial_2) são treinados corretamente,
    se retornam previsões válidas, e se o dicionário de resultados contém as chaves esperadas.
    """
    # Dados simulados
    df = pd.DataFrame({
        "Data": pd.date_range("2023-01-01", periods=20, freq="D"),
        "Fechamento": range(20),
        "Retorno": range(20),
        "MediaMovel_7d": range(20),
        "DesvioPadrao_7d": range(20),
        "TendenciaAlta": [True, False] * 10
    })

    # Executa o treinamento
    resultados: Dict[str, Any] = treinar_modelos(
        df,
        nome_cripto="teste_unitario",
        reutilizar=False,
        num_folds=3
    )

    # Verificações básicas
    for modelo in ["MLP", "Linear", "Polinomial_2"]:
        assert modelo in resultados, f"{modelo} não encontrado nos resultados"

        resultado = resultados[modelo]
        assert "modelo" in resultado
        assert "mse" in resultado and isinstance(resultado["mse"], float)
        assert "y_real" in resultado and isinstance(resultado["y_real"], (list, pd.Series, np.ndarray))
        assert "previsoes" in resultado and isinstance(resultado["previsoes"], (list, pd.Series, np.ndarray))
        assert len(resultado["previsoes"]) == len(resultado["y_real"]), f"{modelo}: tamanho de previsões difere de y_real"
