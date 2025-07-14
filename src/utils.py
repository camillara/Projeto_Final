import os
import matplotlib.pyplot as plt
import joblib
import logging
from typing import Any
import pandas as pd
from src.features import adicionar_features_basicas
import logging
import csv

# === GERAÇÃO DE GRÁFICO DE RETORNO ===
import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_grafico_retorno(df_resultados: pd.DataFrame, modelo: str = "MLP") -> None:
    """
    Gera e salva um gráfico de barras com o retorno percentual por criptomoeda para o modelo especificado.

    Args:
        df_resultados (pd.DataFrame): DataFrame contendo os resultados das simulações.
        modelo (str): Nome do modelo a ser plotado (ex: "MLP", "Linear", "POLINOMIAL_2").
    """
    nome_coluna: str = f"RetornoPercentual_{modelo}"
    if nome_coluna not in df_resultados.columns:
        print(
            f"[AVISO] Coluna '{nome_coluna}' não encontrada. Gráfico de retorno não será gerado."
        )
        return

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(df_resultados["Criptomoeda"], df_resultados[nome_coluna], color="skyblue")
    plt.xlabel(f"Retorno Percentual ({modelo})")
    plt.title(f"Retorno Percentual por Criptomoeda ({modelo})")
    plt.tight_layout()
    nome_arquivo: str = f"figures/retornos_criptos_{modelo}.png"
    plt.savefig(nome_arquivo)
    plt.close()
    print(f"[OK] Gráfico salvo em {nome_arquivo}")


# === SALVAR MODELO ===
def salvar_modelo(modelo: Any, nome: str, pasta: str = "modelos") -> None:
    """
    Salva um modelo treinado no disco usando joblib.

    Args:
        modelo (Any): Objeto do modelo treinado.
        nome (str): Nome do arquivo para salvar (ex: 'BTCUSDT_mlp').
        pasta (str): Caminho da pasta onde salvar o modelo.
    """
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, f"{nome}.joblib")
    joblib.dump(modelo, caminho)
    logging.info(f"[{nome}] Modelo salvo em: {caminho}")


# === CARREGAR MODELO ===
def carregar_modelo(nome: str, pasta: str = "modelos") -> Any:
    """
    Carrega um modelo salvo anteriormente do disco.

    Args:
        nome (str): Nome do arquivo salvo.
        pasta (str): Caminho da pasta onde está salvo o modelo.

    Returns:
        Any: O modelo carregado ou None se não existir.
    """
    caminho = os.path.join(pasta, f"{nome}.joblib")
    if os.path.exists(caminho):
        logging.info(f"[{nome}] Modelo carregado de: {caminho}")
        return joblib.load(caminho)
    return None


# === PREPROCESSAMENTO DE DADOS ===
def preprocessar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função de pré-processamento padrão para os datasets de criptomoedas.

    - Converte colunas para float se necessário.
    - Renomeia 'close' para 'Fechamento' e calcula o 'Retorno'.
    - Remove colunas desnecessárias como 'Unnamed: 0', 'unix', 'symbol', etc.
    - Adiciona features básicas (média móvel, desvio padrão, tendência, etc.)
    - Padroniza nomes de colunas para consistência com os modelos treinados.

    Args:
        df (pd.DataFrame): DataFrame original carregado do CSV.

    Returns:
        pd.DataFrame: DataFrame limpo e pronto para uso.
    """
    df = df.copy()

    logging.info(f"[PREPROCESSAMENTO] Iniciando o pré-processamento...")

    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        df = df.sort_values("Data").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if "close" in df.columns and "Fechamento" not in df.columns:
        df.rename(columns={"close": "Fechamento"}, inplace=True)

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = adicionar_features_basicas(df)

    colunas_validas = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if "Fechamento" in df.columns:
        colunas_validas.append("Fechamento")
    colunas_validas = list(set(colunas_validas))  # remove duplicatas

    logging.info("[PREPROCESSAMENTO] Pré-processamento finalizado com sucesso.")

    return df[colunas_validas + (["Data"] if "Data" in df.columns else [])]


# ===SALVAR MEDIDAS DE DISPERSÃO===
def salvar_medidas_dispersao(nome_cripto, desvio, variancia, amplitude, iqr):
    path = "results/medidas_dispersao.csv"
    headers = ["Criptomoeda", "Desvio Padrao", "Variancia", "Amplitude", "IQR"]

    linha = [nome_cripto, desvio, variancia, amplitude, iqr]

    try:
        with open(path, "x", newline="") as csvfile:  # cria se não existir
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerow(linha)
    except FileExistsError:
        with open(path, "a", newline="") as csvfile:  # adiciona se já existir
            writer = csv.writer(csvfile)
            writer.writerow(linha)
