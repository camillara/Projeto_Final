import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Dict
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.metrics import mean_squared_error

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
    

def plotar_dispersao_e_lucros(resultados: Dict[str, Dict], pasta: str = "figures") -> None:
    """
    Gera três gráficos e arquivos:
    1. Diagrama de dispersão das previsões vs valores reais para todos os modelos.
    2. Gráfico de linha com a evolução do lucro acumulado para cada modelo ao longo do tempo.
    3. Coeficiente de correlação de Pearson para cada modelo (arquivo CSV).
    4. Equações de regressão linear e polinomial para os modelos (arquivo CSV).
    5. Erro padrão para cada modelo (arquivo CSV).

    Args:
        resultados (Dict[str, Dict]): Dicionário contendo os resultados de cada modelo,
                                      com previsões, simulações e métricas.
        pasta (str): Caminho da pasta onde os gráficos e arquivos serão salvos.
    """
    os.makedirs(pasta, exist_ok=True)
    logging.info(f"[Visualização] Pasta '{pasta}' criada para salvar os gráficos.")

    # === Diagrama de Dispersão ===
    plt.figure(figsize=(12, 8))
    for nome_modelo, info in resultados.items():
        if nome_modelo == "MLP":
            y_real = resultados['MLP']['simulacao']['PrecoHoje'].values if 'PrecoHoje' in resultados['MLP']['simulacao'] else resultados['MLP']['simulacao'].get('PrecoHoje', [])
        else:
            y_real = resultados[nome_modelo]['simulacao']['PrecoHoje'].values if 'PrecoHoje' in resultados[nome_modelo]['simulacao'] else resultados[nome_modelo]['simulacao'].get('PrecoHoje', [])
        previsoes = info["previsoes"][:len(y_real)]
        if len(previsoes) != len(y_real) or len(y_real) == 0:
            logging.warning(f"[Dispersão] Modelo '{nome_modelo}' ignorado por dados inconsistentes.")
            continue

        plt.scatter(y_real, previsoes, label=nome_modelo, alpha=0.6)
        logging.info(f"[Dispersão] Modelo '{nome_modelo}' plotado com {len(previsoes)} pontos.")

    plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], 'k--', label="Ideal")
    plt.xlabel("Preço Real")
    plt.ylabel("Preço Previsto")
    plt.title("Diagrama de Dispersão - Preço Real vs Previsto")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    caminho_dispersao = os.path.join(pasta, "dispersao_modelos.png")
    plt.savefig(caminho_dispersao)
    logging.info(f"[Dispersão] Gráfico salvo em: {caminho_dispersao}")
    plt.close()

    # === Gráfico de Lucro Acumulado ===
    plt.figure(figsize=(12, 8))
    for nome_modelo, info in resultados.items():
        df_sim = info["simulacao"]
        if "CapitalFinal" not in df_sim:
            logging.warning(f"[Lucros] Modelo '{nome_modelo}' ignorado por não conter dados de capital.")
            continue

        plt.plot(df_sim["Data"], df_sim["CapitalFinal"], label=nome_modelo)
        logging.info(f"[Lucros] Modelo '{nome_modelo}' plotado com {len(df_sim)} pontos de capital.")

    plt.xlabel("Data")
    plt.ylabel("Capital Final (USD)")
    plt.title("Evolução do Lucro - Simulação de Investimentos")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    caminho_lucro = os.path.join(pasta, "lucros_modelos.png")
    plt.savefig(caminho_lucro)
    logging.info(f"[Lucros] Gráfico salvo em: {caminho_lucro}")
    plt.close()

    # === Coeficientes de Correlação, Equações e Erros Padrão ===
    correlacoes = {}
    equacoes = []
    erros = []

    for nome_modelo, info in resultados.items():
        if 'simulacao' not in info or 'PrecoHoje' not in info['simulacao']:
            logging.warning(f"[Correlação] Modelo '{nome_modelo}' sem dados válidos para cálculo.")
            continue

        y_real = info['simulacao']['PrecoHoje'].values
        previsoes = info['previsoes'][:len(y_real)]

        if len(previsoes) != len(y_real) or len(y_real) == 0:
            logging.warning(f"[Correlação] Modelo '{nome_modelo}' ignorado por dados inconsistentes.")
            continue

        correlacao = np.corrcoef(y_real, previsoes)[0, 1]
        correlacoes[nome_modelo] = correlacao
        logging.info(f"[Correlação] Modelo '{nome_modelo}' coeficiente de correlação: {correlacao:.4f}")

        # === Ajuste de regressão linear ===
        coef_linear = np.polyfit(y_real, previsoes, deg=1)
        eq_linear = f"y = {coef_linear[0]:.4f}x + {coef_linear[1]:.4f}"

        # === Ajuste de regressão polinomial grau 2 a 10 ===
        melhor_r2 = -np.inf
        melhor_eq = ""
        for grau in range(2, 11):
            p = Polynomial.fit(y_real, previsoes, grau)
            r2 = np.corrcoef(previsoes, p(y_real))[0, 1] ** 2
            if r2 > melhor_r2:
                melhor_r2 = r2
                melhor_eq = f"Pol grau {grau}: {p.convert().coef}"

        equacoes.append({
            "Modelo": nome_modelo,
            "Linear": eq_linear,
            "MelhorPolinomial": melhor_eq,
            "R2_Polinomial": round(melhor_r2, 4)
        })

        # === Erro padrão ===
        erro_padrao = np.sqrt(mean_squared_error(y_real, previsoes))
        erros.append({"Modelo": nome_modelo, "ErroPadrao": round(erro_padrao, 4)})
        logging.info(f"[Erro] Modelo '{nome_modelo}' erro padrão: {erro_padrao:.4f}")

    # Salva os coeficientes em CSV
    df_corr = pd.DataFrame(list(correlacoes.items()), columns=['Modelo', 'CoeficienteCorrelacao'])
    caminho_corr = os.path.join(pasta, "coeficientes_correlacao.csv")
    df_corr.to_csv(caminho_corr, index=False)
    logging.info(f"[Correlação] Coeficientes salvos em: {caminho_corr}")

    # Salva as equações de regressão
    df_eq = pd.DataFrame(equacoes)
    caminho_eq = os.path.join(pasta, "equacoes_regressao.csv")
    df_eq.to_csv(caminho_eq, index=False)
    logging.info(f"[Regressão] Equações de regressão salvas em: {caminho_eq}")

    # Salva os erros padrão
    df_erros = pd.DataFrame(erros)
    caminho_erro = os.path.join(pasta, "erros_padrao.csv")
    df_erros.to_csv(caminho_erro, index=False)
    logging.info(f"[Erro] Erros padrão salvos em: {caminho_erro}")

