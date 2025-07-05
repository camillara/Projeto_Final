import os
import pandas as pd
import joblib
from src.utils import plot_grafico_retorno, salvar_modelo, carregar_modelo

def test_plot_grafico_retorno_cria_arquivo():
    df = pd.DataFrame({
        "Criptomoeda": ["BTC", "ETH", "ADA"],
        "RetornoPercentual": [12.5, -3.2, 7.8]
    })

    caminho = "figures/retornos_criptos.png"

    plot_grafico_retorno(df)

    assert os.path.exists(caminho)

    os.remove(caminho)


def test_salvar_e_carregar_modelo():
    modelo_exemplo = {"param": 42, "accuracy": 0.95}
    nome_arquivo = "modelo_teste_utils"
    pasta = "modelos"

    salvar_modelo(modelo_exemplo, nome_arquivo, pasta)

    caminho = os.path.join(pasta, f"{nome_arquivo}.joblib")
    assert os.path.exists(caminho)

    modelo_carregado = carregar_modelo(nome_arquivo, pasta)

    assert modelo_carregado == modelo_exemplo

    os.remove(caminho)
