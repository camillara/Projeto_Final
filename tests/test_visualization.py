import os
import logging
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from src.visualization import (
    salvar_grafico,
    plot_boxplot,
    plot_histograma,
    plot_linha_media_mediana_moda,
    calcular_dispersao,
    plotar_dispersao_e_lucros,
    plot_grafico_comparativo_modelos,
    plot_comparativo_modelos_por_cripto,
    plot_linha_media_mediana_moda,
    moda_rolante,
    salvar_graficos_mlp,
    salvar_graficos_regressao,
    salvar_importancia_features,
    plot_analise_exploratoria_conjunta
)
import pandas as pd


def test_salvar_grafico_cria_arquivo() -> None:
    """
    Testa a função salvar_grafico() garantindo que um gráfico matplotlib é salvo corretamente no disco.
    
    Cria um gráfico simples, invoca a função e verifica se o arquivo foi criado. Depois remove o arquivo.
    """
    nome_arquivo = "grafico_teste"
    pasta_destino = "figures"
    caminho_completo = os.path.join(pasta_destino, f"{nome_arquivo}.png")

    plt.plot([1, 2, 3], [4, 5, 6])

    salvar_grafico(nome_arquivo, pasta_destino)

    assert os.path.exists(caminho_completo)

    os.remove(caminho_completo)
    

def test_plot_boxplot_cria_arquivo() -> None:
    """
    Testa a função plot_boxplot() verificando se o arquivo do gráfico é criado corretamente.
    
    Usa um DataFrame fictício com dados de fechamento, chama a função e verifica se o arquivo foi salvo.
    """
    nome_cripto = "cripto_teste"
    caminho_esperado = os.path.join("figures", f"{nome_cripto}_boxplot.png")

    df = pd.DataFrame({
        "Fechamento": [100, 110, 115, 105, 120, 95, 130]
    })

    plot_boxplot(df, nome_cripto)

    assert os.path.exists(caminho_esperado)

    os.remove(caminho_esperado)
    

def test_plot_histograma_cria_arquivo() -> None:
    """
    Testa a função plot_histograma() verificando se o arquivo do gráfico é criado corretamente.

    Gera um DataFrame sintético com valores de fechamento, chama a função, e verifica se o arquivo foi salvo.
    """
    nome_cripto = "cripto_teste_hist"
    caminho_esperado = os.path.join("figures", f"{nome_cripto}_histograma.png")

    df = pd.DataFrame({
        "Data": pd.date_range(start="2023-01-01", periods=7, freq="D"),
        "Fechamento": [100, 102, 101, 103, 99, 98, 97]
    })

    plot_histograma(df, nome_cripto)

    assert os.path.exists(caminho_esperado)

    os.remove(caminho_esperado)
    

def test_moda_rolante() -> None:
    """
    Testa a função moda_rolante() com diferentes cenários:
    - Série com moda bem definida.
    - Série com todos os valores únicos.
    - Série vazia.
    """
    serie_1 = pd.Series([1, 2, 3, 3, 3, 4])
    assert moda_rolante(serie_1) == 3, "A moda esperada é 3"

    serie_2 = pd.Series([10, 20, 30])
    assert moda_rolante(serie_2) in [10, 20, 30], "Para todos únicos, qualquer valor é aceitável"

    serie_3 = pd.Series([], dtype=float)
    resultado = moda_rolante(serie_3)
    assert np.isnan(resultado), "Para série vazia, o retorno deve ser NaN"
    

def test_calcular_dispersao(caplog) -> None:
    """
    Testa a função calcular_dispersao() com um DataFrame de exemplo.

    Verifica se as medidas de dispersão (desvio padrão, variância, amplitude, IQR)
    são corretamente calculadas e registradas no log.
    """
    df = pd.DataFrame({
        "Fechamento": [10, 12, 14, 16, 18, 20, 22]
    })

    nome_cripto = "TESTE"

    with caplog.at_level(logging.INFO):
        calcular_dispersao(df, nome_cripto)

    assert f"[DISPERSÃO] {nome_cripto}" in caplog.text
    assert "Desvio padrão" in caplog.text
    assert "Variância" in caplog.text
    assert "Amplitude" in caplog.text
    assert "IQR (Q3 - Q1)" in caplog.text
    

def test_plotar_dispersao_e_lucros_gera_arquivos() -> None:
    """
    Testa a função plotar_dispersao_e_lucros verificando se todos os arquivos esperados
    são gerados corretamente: gráfico de dispersão, além dos CSVs de correlação,
    equações de regressão e erro padrão.
    """
    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    from src.visualization import plotar_dispersao_e_lucros

    pasta = "figures"
    hoje = datetime.today()
    datas = pd.date_range(hoje - timedelta(days=9), periods=10)

    resultados_falsos = {
        "MLP": {
            "previsoes": np.linspace(10, 20, 10).tolist(),
            "simulacao": pd.DataFrame({
                "Data": datas,
                "CapitalFinal": np.linspace(1000, 2000, 10),
                "PrecoHoje": np.linspace(10, 20, 10)
            })
        },
        "Linear": {
            "previsoes": np.linspace(12, 22, 10).tolist(),
            "simulacao": pd.DataFrame({
                "Data": datas,
                "CapitalFinal": np.linspace(1100, 1900, 10),
                "PrecoHoje": np.linspace(10, 20, 10)
            })
        }
    }

    plotar_dispersao_e_lucros(resultados_falsos, pasta=pasta)

    arquivos_esperados = [
        f"{pasta}/dispersao_modelos.png",
        f"{pasta}/coeficientes_correlacao.csv",
        f"{pasta}/equacoes_regressao.csv",
        f"{pasta}/erros_padrao.csv"
    ]

    for caminho in arquivos_esperados:
        assert os.path.exists(caminho), f"Arquivo não encontrado: {caminho}"

    for caminho in arquivos_esperados:
        os.remove(caminho)

        

def test_plot_grafico_comparativo_modelos() -> None:
    """
    Testa a função plot_grafico_comparativo_modelos verificando se o gráfico é gerado
    corretamente com um DataFrame de exemplo e salvo no local esperado.
    """
    df_resultados = pd.DataFrame({
        "Criptomoeda": ["BTC", "ETH", "XRP"],
        "RetornoPercentual_MLP": [12.5, 8.3, None],
        "RetornoPercentual_Linear": [10.1, None, 5.0],
        "RetornoPercentual_Polinomial_2": [11.2, 9.8, 6.1]
    })

    caminho_grafico = "figures/retorno_modelos_comparativo.png"

    if os.path.exists(caminho_grafico):
        os.remove(caminho_grafico)

    plot_grafico_comparativo_modelos(df_resultados)

    assert os.path.exists(caminho_grafico), f"Gráfico não foi criado: {caminho_grafico}"

    os.remove(caminho_grafico)
    
    
def test_plot_comparativo_modelos_por_cripto() -> None:
    """
    Testa a função plot_grafico_comparativo_modelos que gera um gráfico de barras
    para cada criptomoeda comparando os retornos de todos os modelos.
    """

    df = pd.DataFrame({
        "Criptomoeda": ["BTC", "ETH"],
        "RetornoPercentual_MLP": [10.5, 9.3],
        "RetornoPercentual_Linear": [8.2, 7.9],
        "RetornoPercentual_Polinomial_2": [9.8, 8.5],
        "RetornoPercentual_Polinomial_3": [9.9, 8.8],
        "RetornoPercentual_Polinomial_4": [10.0, 8.9],
        "RetornoPercentual_Polinomial_5": [10.1, 9.0],
        "RetornoPercentual_Polinomial_6": [10.2, 9.1],
        "RetornoPercentual_Polinomial_7": [10.3, 9.2],
        "RetornoPercentual_Polinomial_8": [10.4, 9.3],
        "RetornoPercentual_Polinomial_9": [10.5, 9.4],
        "RetornoPercentual_Polinomial_10": [10.6, 9.5]
    })

    pasta = "figures/modelos_por_cripto"
    arquivos_esperados = [os.path.join(pasta, f"{cripto}_modelos.png") for cripto in df["Criptomoeda"]]


    for arquivo in arquivos_esperados:
        if os.path.exists(arquivo):
            os.remove(arquivo)

    plot_comparativo_modelos_por_cripto(df)

    for arquivo in arquivos_esperados:
        assert os.path.exists(arquivo), f"Gráfico não foi gerado: {arquivo}"
        os.remove(arquivo)
        

def test_plot_histograma_cria_arquivo() -> None:
    """
    Testa se a função `plot_histograma` salva corretamente o gráfico de histograma
    em um arquivo na pasta 'figures'.
    """
    nome_cripto: str = "cripto_histograma"
    caminho_esperado: str = os.path.join("figures", f"{nome_cripto}_histograma.png")

    df: pd.DataFrame = pd.DataFrame({"Fechamento": np.random.normal(100, 10, 100)})
    plot_histograma(df, nome_cripto)

    assert os.path.exists(caminho_esperado)
    os.remove(caminho_esperado)
    

def test_plot_linha_media_mediana_moda_cria_arquivo() -> None:
    """
    Testa se o gráfico com média, mediana e moda é salvo corretamente.
    """
    nome_cripto = "cripto_linha"
    caminho_esperado = os.path.join("figures", f"{nome_cripto}_linha_tempo.png")

    df = pd.DataFrame({
        "Data": pd.date_range(start="2023-01-01", periods=7, freq="D"),
        "Fechamento": [100, 102, 101, 103, 99, 98, 97]
    })

    plot_linha_media_mediana_moda(df, nome_cripto)

    assert os.path.exists(caminho_esperado)

    os.remove(caminho_esperado)



def test_plotar_dispersao_e_lucros_cria_graficos() -> None:
    """
    Testa se `plotar_dispersao_e_lucros` salva corretamente os gráficos de dispersão
    e evolução de lucros na pasta figures/dispersao/<nome_cripto>.
    """

    nome_cripto: str = "cripto_dispersao"
    caminho_saida: str = os.path.join("figures", "dispersao", nome_cripto)
    os.makedirs(caminho_saida, exist_ok=True)

    resultados: dict[str, dict[str, Union[np.ndarray, pd.DataFrame]]] = {
        "ModeloTeste": {
            "previsoes": np.array([100, 105, 110, 108]),
            "reais": np.array([102, 104, 109, 107]),
            "simulacao": pd.DataFrame({
                "PrecoHoje": [102, 104, 109, 107],
                "PrecoPrevisto": [100, 105, 110, 108]
            })
        }
    }

    plotar_dispersao_e_lucros(resultados, caminho_saida)

    arquivos: list[str] = os.listdir(caminho_saida)

    assert "dispersao_modelos.png" in arquivos
    assert "coeficientes_correlacao.csv" in arquivos
    assert "equacoes_regressao.csv" in arquivos
    assert "erros_padrao.csv" in arquivos

    for nome in arquivos:
        os.remove(os.path.join(caminho_saida, nome))
    os.rmdir(caminho_saida)
    

def criar_mock_dataframe() -> pd.DataFrame:
    """
    Cria um DataFrame de exemplo com colunas 'Data' e 'Fechamento'.

    Returns:
        pd.DataFrame: DataFrame com dados simulados.
    """
    return pd.DataFrame({
        "Data": pd.date_range(start="2023-01-01", periods=10, freq="D"),
        "Fechamento": np.linspace(100, 110, 10)
    })


def test_salvar_graficos_mlp() -> None:
    """
    Testa se os gráficos de dispersão e curva de perda do MLP são salvos corretamente.
    """
    y_real = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    loss_curve = [0.5, 0.3, 0.2, 0.15]
    nome = "cripto_teste"

    salvar_graficos_mlp(y_real, y_pred, loss_curve, nome)

    pasta = os.path.join("figures", "MLP", nome)
    assert os.path.exists(os.path.join(pasta, "dispersao_real_vs_previsto.png"))
    assert os.path.exists(os.path.join(pasta, "curva_loss_mlp.png"))


def test_salvar_importancia_features() -> None:
    """
    Testa se o gráfico de importância das features é salvo corretamente.
    """
    X = np.random.rand(100, 4)
    y = X @ np.array([0.4, 0.3, 0.2, 0.1]) + np.random.normal(0, 0.01, 100)
    model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
    model.fit(X, y)
    nome = "cripto_importancia"
    salvar_importancia_features(model, X, y, ["f1", "f2", "f3", "f4"], nome)
    caminho = os.path.join("figures", "MLP", nome, f"importancia_features_{nome}.png")
    assert os.path.exists(caminho)


def test_salvar_graficos_regressao() -> None:
    """
    Testa se o gráfico de dispersão para modelos de regressão não-MLP é salvo corretamente.
    """
    nome_modelo = "Linear"
    nome_cripto = "cripto_linear"
    y_real = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([12, 18, 31, 39, 49])
    salvar_graficos_regressao(nome_modelo, y_real, y_pred, nome_cripto)
    caminho = f"figures/{nome_modelo}/{nome_cripto}/dispersao_real_vs_previsto_{nome_modelo}_{nome_cripto}.png"
    assert os.path.exists(caminho)


def test_plot_analise_exploratoria_conjunta() -> None:
    """
    Testa se o gráfico da análise exploratória conjunta é salvo corretamente.
    """
    nome = "cripto_exploratoria"
    df = criar_mock_dataframe()
    plot_analise_exploratoria_conjunta(df, nome)
    caminho = os.path.join("figures", f"{nome}_analise_exploratoria.png")
    assert os.path.exists(caminho)