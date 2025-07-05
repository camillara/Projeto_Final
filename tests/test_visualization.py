import os
import pandas as pd
from src.visualization import (
    plot_boxplot,
    plot_histograma,
    plot_linha_media_mediana_moda,
    calcular_dispersao
)

def test_plot_boxplot_cria_arquivo():
    df = pd.DataFrame({'Fechamento': [1, 2, 2, 3, 4]})
    nome_cripto = "teste_boxplot"
    caminho = os.path.join("figures", f"{nome_cripto}_boxplot.png")

    if os.path.exists(caminho):
        os.remove(caminho)

    plot_boxplot(df, nome_cripto)

    assert os.path.exists(caminho)

def test_plot_histograma_cria_arquivo():
    df = pd.DataFrame({'Fechamento': [1, 2, 2, 3, 3, 3]})
    nome_cripto = "teste_histograma"
    caminho = os.path.join("figures", f"{nome_cripto}_histograma.png")

    if os.path.exists(caminho):
        os.remove(caminho)

    plot_histograma(df, nome_cripto)

    assert os.path.exists(caminho)

def test_plot_linha_media_mediana_moda_cria_arquivo():
    df = pd.DataFrame({
        'Data': pd.date_range(start="2023-01-01", periods=10),
        'Fechamento': [10, 11, 10, 12, 13, 11, 12, 13, 14, 15]
    })
    nome_cripto = "teste_linha_tempo"
    caminho = os.path.join("figures", f"{nome_cripto}_linha_tempo.png")

    if os.path.exists(caminho):
        os.remove(caminho)

    plot_linha_media_mediana_moda(df, nome_cripto)

    assert os.path.exists(caminho)

def test_calcular_dispersao_nao_lanca_excecao():
    df = pd.DataFrame({'Fechamento': [10, 12, 11, 13, 15]})
    nome_cripto = "teste_dispersao"

    
    calcular_dispersao(df, nome_cripto)
