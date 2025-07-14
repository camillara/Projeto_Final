import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Dict
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from src.utils import salvar_medidas_dispersao

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
    salvar_medidas_dispersao(nome_cripto, desvio_padrao, variancia, amplitude, iqr)


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
    plt.savefig(caminho_dispersao, dpi = 150)
    logging.info(f"[Dispersão] Gráfico salvo em: {caminho_dispersao}")
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
    
    

def plot_grafico_comparativo_modelos(df_resultados):
    """
    Gera um gráfico de barras comparando o retorno percentual obtido por diferentes modelos
    (MLP, Regressão Linear e Polinomial de Grau 2) para cada criptomoeda analisada.

    O gráfico é salvo em 'figures/retorno_modelos_comparativo.png'.

    Parâmetros:
    -----------
    df_resultados : pandas.DataFrame
        DataFrame contendo as colunas:
        - 'Criptomoeda'
        - 'RetornoPercentual_MLP'
        - 'RetornoPercentual_Linear'
        - 'RetornoPercentual_Poly_2'

    Retorno:
    --------
    None
    """

    # Extrai os nomes das criptomoedas e os retornos dos modelos, substituindo None por 0
    criptos = df_resultados['Criptomoeda']
    mlp = df_resultados['RetornoPercentual_MLP'].fillna(0)
    linear = df_resultados['RetornoPercentual_Linear'].fillna(0)
    poly2 = df_resultados['RetornoPercentual_Polinomial_2'].fillna(0)

    # Define o número de barras e o espaçamento entre elas
    x = np.arange(len(criptos))
    largura_barra = 0.25

    # Cria a figura e o eixo do gráfico
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plota as barras de cada modelo deslocadas horizontalmente
    ax.bar(x - largura_barra, mlp, width=largura_barra, label='MLP', color='skyblue')
    ax.bar(x, linear, width=largura_barra, label='Linear', color='orange')
    ax.bar(x + largura_barra, poly2, width=largura_barra, label='Polinomial Grau 2', color='green')

    # Ajustes do eixo x
    ax.set_xticks(x)
    ax.set_xticklabels(criptos, rotation=45, ha='right')

    # Rótulos e título
    ax.set_ylabel('Retorno Percentual')
    ax.set_title('Comparação de Retorno Percentual por Modelo e Criptomoeda')

    # Exibe legenda e grade
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # Ajusta layout para não cortar elementos
    plt.tight_layout()

    # Garante que o diretório de saída exista
    os.makedirs("figures", exist_ok=True)

    # Salva o gráfico como imagem
    plt.savefig("figures/retorno_modelos_comparativo.png")
    plt.close()

    # Confirmação no console
    print("[OK] Gráfico comparativo salvo em figures/retorno_modelos_comparativo.png")


def plot_comparativo_modelos_por_cripto(df_resultados: pd.DataFrame):
    """
    Gera um gráfico de barras para cada criptomoeda comparando o retorno percentual
    de todos os modelos (MLP, Linear, Polinomiais grau 2 a 10).

    Os gráficos são salvos em 'figures/modelos_por_cripto/<cripto>.png'.

    Parâmetros:
    -----------
    df_resultados : pandas.DataFrame
        DataFrame contendo colunas:
        - 'Criptomoeda'
        - 'RetornoPercentual_MLP'
        - 'RetornoPercentual_Linear'
        - 'RetornoPercentual_Polinomial_2' até 'RetornoPercentual_Polinomial_10'

    Retorno:
    --------
    None
    """

    # Garante que o diretório de saída exista
    pasta_saida = "figures/modelos_por_cripto"
    os.makedirs(pasta_saida, exist_ok=True)

    # Para cada criptomoeda, gera um gráfico individual
    for _, row in df_resultados.iterrows():
        cripto = row['Criptomoeda']

        # Coleta os retornos disponíveis para os modelos
        modelos = ['MLP', 'Linear'] + [f'Polinomial_{i}' for i in range(2, 11)]
        colunas = [f'RetornoPercentual_{m}' for m in modelos]
        retornos = [row.get(col, 0) or 0 for col in colunas]  # trata None como 0

        # Cria gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(modelos))

        ax.bar(x, retornos, color='skyblue')
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=45, ha='right')
        ax.set_ylabel('Retorno Percentual')
        ax.set_title(f'Retorno por Modelo - {cripto}')
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        caminho_arquivo = os.path.join(pasta_saida, f"{cripto}_modelos.png")
        plt.savefig(caminho_arquivo)
        plt.close()

        print(f"[OK] Gráfico salvo: {caminho_arquivo}")
        

def salvar_graficos_mlp(y_real, y_pred, loss_curve, nome_cripto):
    """
    Salva dois gráficos:
    - Dispersão entre valores reais e previstos
    - Curva de perda (loss) durante o treinamento
    """
    pasta = os.path.join("figures", "MLP", nome_cripto)
    os.makedirs(pasta, exist_ok=True)

    # Gráfico 1: Dispersão Real vs Previsto
    plt.figure(figsize=(8, 6))
    plt.scatter(y_real, y_pred, alpha=0.6, color="purple", edgecolor="k")
    plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], color="red", linestyle="--")
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.title(f"Dispersão: Real vs Previsto (MLP - {nome_cripto})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, "dispersao_real_vs_previsto.png"), dpi=150)
    plt.close()

    # Gráfico 2: Curva de perda durante o treinamento
    plt.figure(figsize=(8, 6))
    plt.plot(loss_curve, marker='o')
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title(f"Curva de Treinamento (Loss) - MLP ({nome_cripto})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, "curva_loss_mlp.png"), dpi=150)
    plt.close()

    print(f"[OK] Gráficos do MLP salvos em {pasta}")
    

def salvar_importancia_features(
    modelo: MLPRegressor,
    X_scaled: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    nome_cripto: str
) -> None:
    """
    Gera e salva gráfico de importância das features com Permutation Importance.

    Args:
        modelo (MLPRegressor): Modelo treinado.
        X_scaled (np.ndarray): Dados de entrada escalonados.
        y (np.ndarray): Valores reais.
        feature_names (list[str]): Lista de nomes das features.
        nome_cripto (str): Nome da criptomoeda (para nome do arquivo).
        pasta_destino (str): Pasta onde salvar o gráfico.
    """
    resultados = permutation_importance(modelo, X_scaled, y, n_repeats=10, random_state=42)

    importancias = resultados.importances_mean
    indices = np.argsort(importancias)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importancias)), importancias[indices], align="center")
    plt.xticks(range(len(importancias)), np.array(feature_names)[indices], rotation=90)
    plt.title(f"Importância das Features (MLP - {nome_cripto})")
    plt.tight_layout()

    pasta = os.path.join("figures", "MLP", nome_cripto)
    os.makedirs(pasta, exist_ok=True)
    path = os.path.join(pasta, f"importancia_features_{nome_cripto}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"[OK] Gráfico de importância das features salvo em {path}")
    

def salvar_graficos_regressao(nome_modelo: str, y_real, y_pred, nome_cripto: str):
    """
    Gera e salva o gráfico de dispersão com linha de referência para modelos de regressão não-MLP.
    """
    os.makedirs(f"figures/{nome_modelo}/{nome_cripto}", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_real, y_pred, alpha=0.5)
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.title(f"{nome_modelo}: Real vs Previsto ({nome_cripto})")

    # Adiciona linha de referência y = x (linha ideal)
    min_val = min(np.min(y_real), np.min(y_pred))
    max_val = max(np.max(y_real), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)

    plt.grid(True)
    plt.tight_layout()

    path = f"figures/{nome_modelo}/{nome_cripto}/dispersao_real_vs_previsto_{nome_modelo}_{nome_cripto}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"[OK] Gráfico de dispersão salvo em {path}")
    

def plot_analise_exploratoria_conjunta(df: pd.DataFrame, nome_cripto: str) -> None:
    """
    Gera e salva uma figura com 4 subplots:
      1. Boxplot do fechamento
      2. Histograma do fechamento
      3. Linha do fechamento com média, mediana e moda móveis
      4. Texto com medidas de dispersão: desvio, variância, amplitude e IQR.

    Args:
        df (pd.DataFrame): DataFrame contendo colunas 'Data' e 'Fechamento'.
        nome_cripto (str): Nome da criptomoeda (para nome do arquivo e título).
    """
    # calcula as medidas de dispersão
    fechamento = df['Fechamento'].dropna()
    desvio = fechamento.std()
    variancia = fechamento.var()
    amplitude = fechamento.max() - fechamento.min()
    q1, q3 = fechamento.quantile(0.25), fechamento.quantile(0.75)
    iqr = q3 - q1

    # prepara figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.set_style("whitegrid")

    # 1) Boxplot
    sns.boxplot(y=fechamento, ax=axes[0, 0])
    axes[0, 0].set_title(f"{nome_cripto} — Boxplot de Fechamento")
    axes[0, 0].set_ylabel("Preço de Fechamento")

    # 2) Histograma
    sns.histplot(fechamento, bins=30, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title(f"{nome_cripto} — Histograma de Fechamento")
    axes[0, 1].set_xlabel("Preço de Fechamento")

    # 3) Linha com média, mediana e moda (7d)
    dfm = df.copy()
    dfm['Media']   = dfm['Fechamento'].rolling(7).mean()
    dfm['Mediana'] = dfm['Fechamento'].rolling(7).median()
    # usa sua função de moda rolante
    from src.visualization import moda_rolante
    dfm['Moda']    = dfm['Fechamento'].rolling(7).apply(moda_rolante, raw=False)

    ax = axes[1, 0]
    ax.plot(dfm['Data'], dfm['Fechamento'], label='Fechamento', lw=1)
    ax.plot(dfm['Data'], dfm['Media'],   label='Média (7d)',  ls='--')
    ax.plot(dfm['Data'], dfm['Mediana'], label='Mediana (7d)',ls='-.')
    ax.plot(dfm['Data'], dfm['Moda'],    label='Moda (7d)',   ls=':')
    ax.set_title(f"{nome_cripto} — Fechamento e Médias Móveis")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço")
    ax.legend()

    # 4) Texto com dispersão
    ax = axes[1, 1]
    ax.axis('off')
    text = (
        f"Desvio Padrão: {desvio:.4f}\n"
        f"Variância:       {variancia:.4f}\n"
        f"Amplitude:       {amplitude:.4f}\n"
        f"IQR (Q3–Q1):     {iqr:.4f}"
    )
    ax.text(0.1, 0.5, text, fontsize=12, family='monospace')
    ax.set_title(f"{nome_cripto} — Medidas de Dispersão")

    plt.tight_layout()
    pasta = "figures"
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, f"{nome_cripto}_analise_exploratoria.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    logging.info(f"[GRAFICO] Analise exploratória salva em: {caminho}")
