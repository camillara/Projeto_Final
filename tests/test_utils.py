import os
import pandas as pd
import joblib
from typing import Dict, Any
from src.utils import plot_grafico_retorno, salvar_modelo, carregar_modelo


def test_plot_grafico_retorno_cria_arquivo() -> None:
    """
    Testa a função plot_grafico_retorno() verificando se o arquivo de gráfico é criado corretamente.

    O gráfico é gerado a partir de dados fictícios de retorno percentual de criptomoedas,
    e ao final o arquivo gerado é removido.
    """
    df_fake = pd.DataFrame({
        "Criptomoeda": ["BTC", "ETH", "ADA"],
        "RetornoPercentual_MLP": [12.5, -3.2, 7.8]
    })

    caminho_arquivo = "figures/retornos_criptos.png"

    # Executa a função de plotagem
    plot_grafico_retorno(df_fake)

    # Verifica se o arquivo foi criado
    assert os.path.exists(caminho_arquivo)

    # Limpeza
    os.remove(caminho_arquivo)


def test_salvar_e_carregar_modelo() -> None:
    """
    Testa as funções salvar_modelo() e carregar_modelo().

    Garante que um dicionário de exemplo salvo em disco seja carregado corretamente,
    e que o conteúdo carregado seja igual ao original.
    """
    modelo_exemplo: Dict[str, Any] = {"param": 42, "accuracy": 0.95}
    nome_arquivo = "modelo_teste_utils"
    pasta_destino = "modelos"
    caminho_modelo = os.path.join(pasta_destino, f"{nome_arquivo}.joblib")

    # Salva o modelo
    salvar_modelo(modelo_exemplo, nome_arquivo, pasta_destino)

    # Verifica se o arquivo foi salvo
    assert os.path.exists(caminho_modelo)

    # Carrega e compara
    modelo_carregado = carregar_modelo(nome_arquivo, pasta_destino)
    assert modelo_carregado == modelo_exemplo

    # Limpeza
    os.remove(caminho_modelo)
