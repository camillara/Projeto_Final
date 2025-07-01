import os
import matplotlib.pyplot as plt
import joblib
import logging
from typing import Any

# === GERAÇÃO DE GRÁFICO DE RETORNO ===
def plot_grafico_retorno(df_resultados):
    """
    Gera e salva um gráfico de barras com o retorno percentual por criptomoeda.

    Args:
        df_resultados (pd.DataFrame): DataFrame com colunas 'Criptomoeda' e 'RetornoPercentual'.
    """
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(df_resultados["Criptomoeda"], df_resultados["RetornoPercentual"], color="skyblue")
    plt.xlabel("Retorno Percentual")
    plt.title("Retorno por Criptomoeda (Simulação MLP)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("figures/retornos_criptos.png")
    plt.close()


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
