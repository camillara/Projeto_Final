import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional

# Evita reconfigurar o logging se ele já estiver configurado
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def salvar_grafico(nome: str, pasta: str = "figures", dpi: int = 150) -> None:
    """
    Salva o gráfico atual na pasta especificada com o nome fornecido.

    Args:
        nome (str): Nome do arquivo (sem extensão).
        pasta (str): Nome da pasta onde o arquivo será salvo.
        dpi (int): Qualidade da imagem.
    """
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, f"{nome}.png")
    plt.savefig(caminho, dpi=dpi, bbox_inches="tight")
    plt.close()
    logging.info(f"[GRAFICO] Gráfico salvo em: {caminho}")


def plot_boxplot(df: pd.DataFrame, nome_cripto: str) -> None:
    """
    Gera e salva um boxplot para os preços de fechamento.

    Args:
        df (pd.DataFrame): DataFrame contendo a coluna 'Fechamento'.
        nome_cripto (str): Nome da criptomoeda.
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df['Fechamento'])
    plt.title(f"Boxplot - {nome_cripto}")
    plt.ylabel("Preço de Fechamento")
    salvar_grafico(f"{nome_cripto}_boxplot")


def plot_histograma(df: pd.DataFrame, nome_cripto: str) -> None:
    """
    Gera e salva um histograma para os preços de fechamento.

    Args:
        df (pd.DataFrame): DataFrame contendo a coluna 'Fechamento'.
        nome_cripto (str): Nome da criptomoeda.
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(df['Fechamento'], bins=30, kde=True)
    plt.title(f"Histograma - {nome_cripto}")
    plt.xlabel("Preço de Fechamento")
    plt.ylabel("Frequência")
    salvar_grafico(f"{nome_cripto}_histograma")


def moda_rolante(series: pd.Series) -> Optional[float]:
    """
    Calcula a moda de uma série com fallback para NaN se vazia.

    Args:
        series (pd.Series): Janela da série temporal.

    Returns:
        float | None: Valor da moda ou NaN.
    """
    try:
        moda = stats.mode(series, keepdims=True).mode[0]
        return moda
    except Exception as e:
        logging.warning(f"Erro ao calcular moda rolante: {e}")
        return float('nan')
    
def calcular_dispersao(df: pd.DataFrame, nome_cripto: str) -> None:
    """
    Calcula e exibe as medidas de dispersão para a criptomoeda.

    Args:
        df (pd.DataFrame): DataFrame com a coluna 'Fechamento'.
        nome_cripto (str): Nome da criptomoeda.
    """
    fechamento = df['Fechamento'].dropna()
    desvio_padrao = fechamento.std()
    variancia = fechamento.var()
    amplitude = fechamento.max() - fechamento.min()
    q1 = fechamento.quantile(0.25)
    q3 = fechamento.quantile(0.75)
    iqr = q3 - q1

    logging.info(f"[DISPERSÃO] {nome_cripto}")
    logging.info(f"  Desvio padrão: {desvio_padrao:.4f}")
    logging.info(f"  Variância: {variancia:.4f}")
    logging.info(f"  Amplitude: {amplitude:.4f}")
    logging.info(f"  IQR (Q3 - Q1): {iqr:.4f}")


def plot_linha_media_mediana_moda(df: pd.DataFrame, nome_cripto: str) -> None:
    """
    Gera e salva gráfico com a linha do preço de fechamento,
    média, mediana e moda móveis (7 dias).

    Args:
        df (pd.DataFrame): DataFrame contendo 'Data' e 'Fechamento'.
        nome_cripto (str): Nome da criptomoeda.
    """
    df = df.copy()
    df['Media'] = df['Fechamento'].rolling(window=7).mean()
    df['Mediana'] = df['Fechamento'].rolling(window=7).median()
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
