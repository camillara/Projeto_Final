import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.visualization import (
    salvar_grafico,
    plot_boxplot,
    plot_histograma,
    plot_linha_media_mediana_moda,
    calcular_dispersao,
    plotar_dispersao_e_lucros,
    plot_grafico_comparativo_modelos,
    plot_comparativo_modelos_por_cripto
)

def test_salvar_grafico_cria_arquivo() -> None:
    """
    Testa a função salvar_grafico() garantindo que um gráfico matplotlib é salvo corretamente no disco.
    
    Cria um gráfico simples, invoca a função e verifica se o arquivo foi criado. Depois remove o arquivo.
    """
    nome_arquivo = "grafico_teste"
    pasta_destino = "figures"
    caminho_completo = os.path.join(pasta_destino, f"{nome_arquivo}.png")

    # Cria um gráfico simples antes de chamar salvar_grafico
    plt.plot([1, 2, 3], [4, 5, 6])

    salvar_grafico(nome_arquivo, pasta_destino)

    assert os.path.exists(caminho_completo)

    # Limpeza
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

    # Limpeza
    os.remove(caminho_esperado)
    


def test_plot_histograma_cria_arquivo() -> None:
    """
    Testa a função plot_histograma() verificando se o arquivo do gráfico é criado corretamente.

    Gera um DataFrame sintético com valores de fechamento, chama a função, e verifica se o arquivo foi salvo.
    """
    nome_cripto = "cripto_teste_hist"
    caminho_esperado = os.path.join("figures", f"{nome_cripto}_histograma.png")

    df = pd.DataFrame({
        "Fechamento": [100, 102, 104, 98, 101, 103, 105, 107, 106, 99]
    })

    plot_histograma(df, nome_cripto)

    assert os.path.exists(caminho_esperado)

    # Limpeza
    os.remove(caminho_esperado)
    

import pandas as pd
import numpy as np
from src.visualization import moda_rolante

def test_moda_rolante() -> None:
    """
    Testa a função moda_rolante() com diferentes cenários:
    - Série com moda bem definida.
    - Série com todos os valores únicos.
    - Série vazia.
    """
    # Moda bem definida (3 ocorre mais vezes)
    serie_1 = pd.Series([1, 2, 3, 3, 3, 4])
    assert moda_rolante(serie_1) == 3, "A moda esperada é 3"

    # Todos valores únicos (espera-se o menor valor dentre os empatados, por padrão do scipy)
    serie_2 = pd.Series([10, 20, 30])
    assert moda_rolante(serie_2) in [10, 20, 30], "Para todos únicos, qualquer valor é aceitável"

    # Série vazia (esperado: NaN)
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
    

def test_plot_linha_media_mediana_moda_cria_arquivo() -> None:
    """
    Testa se a função plot_linha_media_mediana_moda() gera e salva corretamente o gráfico.

    Cria um DataFrame fictício com dados suficientes para calcular média, mediana e moda móveis
    e verifica se o gráfico é salvo na pasta figures com o nome esperado.
    """
    df = pd.DataFrame({
        "Data": pd.date_range("2023-01-01", periods=10, freq="D"),
        "Fechamento": [10, 12, 11, 13, 15, 14, 16, 17, 18, 19]
    })
    nome_cripto = "TESTE_MODA"
    caminho = f"figures/{nome_cripto}_linha_tempo.png"

    # Gera gráfico
    plot_linha_media_mediana_moda(df, nome_cripto)

    # Verifica se o arquivo foi criado
    assert os.path.exists(caminho)

    # Limpa o arquivo gerado após o teste
    os.remove(caminho)


def test_plotar_dispersao_e_lucros_gera_arquivos() -> None:
    """
    Testa a função plotar_dispersao_e_lucros verificando se todos os arquivos esperados
    são gerados corretamente: gráficos de dispersão e lucro, além dos CSVs de correlação,
    equações de regressão e erro padrão.
    """
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

    # Executa a função
    plotar_dispersao_e_lucros(resultados_falsos, pasta=pasta)

    # Lista de caminhos esperados
    arquivos_esperados = [
        f"{pasta}/dispersao_modelos.png",
        f"{pasta}/lucros_modelos.png",
        f"{pasta}/coeficientes_correlacao.csv",
        f"{pasta}/equacoes_regressao.csv",
        f"{pasta}/erros_padrao.csv"
    ]

    # Verifica se todos os arquivos foram criados
    for caminho in arquivos_esperados:
        assert os.path.exists(caminho), f"Arquivo não encontrado: {caminho}"

    # Limpa os arquivos após o teste
    for caminho in arquivos_esperados:
        os.remove(caminho)
        


def test_plot_grafico_comparativo_modelos() -> None:
    """
    Testa a função plot_grafico_comparativo_modelos verificando se o gráfico é gerado
    corretamente com um DataFrame de exemplo e salvo no local esperado.
    """
    # Cria um DataFrame simulado
    df_resultados = pd.DataFrame({
        "Criptomoeda": ["BTC", "ETH", "XRP"],
        "RetornoPercentual_MLP": [12.5, 8.3, None],
        "RetornoPercentual_Linear": [10.1, None, 5.0],
        "RetornoPercentual_Polinomial_2": [11.2, 9.8, 6.1]
    })

    # Caminho esperado do gráfico
    caminho_grafico = "figures/retorno_modelos_comparativo.png"

    # Remove o arquivo se já existir
    if os.path.exists(caminho_grafico):
        os.remove(caminho_grafico)

    # Executa a função
    plot_grafico_comparativo_modelos(df_resultados)

    # Verifica se o arquivo foi criado
    assert os.path.exists(caminho_grafico), f"Gráfico não foi criado: {caminho_grafico}"

    # Limpeza após o teste
    os.remove(caminho_grafico)
    
    
def test_plot_comparativo_modelos_por_cripto() -> None:
    """
    Testa a função plot_grafico_comparativo_modelos que gera um gráfico de barras
    para cada criptomoeda comparando os retornos de todos os modelos.
    """
    # DataFrame de exemplo
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

    # Remove arquivos antigos se existirem
    for arquivo in arquivos_esperados:
        if os.path.exists(arquivo):
            os.remove(arquivo)

    # Executa a função
    plot_comparativo_modelos_por_cripto(df)

    # Verifica se os gráficos foram gerados
    for arquivo in arquivos_esperados:
        assert os.path.exists(arquivo), f"Gráfico não foi gerado: {arquivo}"
        os.remove(arquivo)  # Limpa após o teste