import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats


def salvar_grafico(nome: str, pasta: str = "figures", dpi: int = 150) -> None:
    """
    Salva o gráfico atual na pasta especificada.

    Args:
        nome (str): Nome do arquivo de saída (sem extensão).
        pasta (str): Pasta onde salvar os gráficos.
        dpi (int): Resolução da imagem.
    """
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, f"{nome}.png")
    plt.savefig(caminho, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_boxplot(df: pd.DataFrame, nome_cripto: str) -> None:
    """
    Gera e salva o boxplot do preço de fechamento.

    Args:
        df (pd.DataFrame): DataFrame da criptomoeda.
        nome_cripto (str): Nome para o título e nome do arquivo.
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df['Fechamento'])
    plt.title(f"Boxplot - {nome_cripto}")
    plt.ylabel("Preço de Fechamento")
    salvar_grafico(f"{nome_cripto}_boxplot")


def plot_histograma(df: pd.DataFrame, nome_cripto: str) -> None:
    """
    Gera e salva o histograma do preço de fechamento.

    Args:
        df (pd.DataFrame): DataFrame da criptomoeda.
        nome_cripto (str): Nome para o título e nome do arquivo.
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(df['Fechamento'], bins=30, kde=True)
    plt.title(f"Histograma - {nome_cripto}")
    plt.xlabel("Preço de Fechamento")
    plt.ylabel("Frequência")
    salvar_grafico(f"{nome_cripto}_histograma")


def plot_linha_media_mediana_moda(df: pd.DataFrame, nome_cripto: str) -> None:
    """
    Gera e salva gráfico de linha com média, mediana e moda.

    Args:
        df (pd.DataFrame): DataFrame da criptomoeda.
        nome_cripto (str): Nome para o título e nome do arquivo.
    """
    df = df.copy()
    df['Media'] = df['Fechamento'].rolling(window=7).mean()
    df['Mediana'] = df['Fechamento'].rolling(window=7).median()

    # Moda mais frequente no intervalo de 7 dias
    def moda_rolante(series):
        return stats.mode(series, keepdims=True).mode[0]

    df['Moda'] = df['Fechamento'].rolling(window=7).apply(moda_rolante, raw=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df['Data'], df['Fechamento'], label='Fechamento', linewidth=1)
    plt.plot(df['Data'], df['Media'], label='Média (7d)', linestyle='--')
    plt.plot(df['Data'], df['Mediana'], label='Mediana (7d)', linestyle='-.')
    plt.plot(df['Data'], df['Moda'], label='Moda (7d)', linestyle=':')

    plt.title(f"Preço de Fechamento ao Longo do Tempo - {nome_cripto}")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.legend()
    salvar_grafico(f"{nome_cripto}_linha_tempo")
