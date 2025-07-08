import os
import matplotlib.pyplot as plt
import joblib
import logging
from typing import Any

# === GERAÇÃO DE GRÁFICO DE RETORNO ===
def plot_grafico_retorno(df_resultados):
    """
    Gera e salva um gráfico de barras com o retorno percentual por criptomoeda.
    Usa a coluna 'RetornoPercentual_MLP' como padrão, já que representa a simulação do modelo MLP.

    Args:
        df_resultados (pd.DataFrame): DataFrame com colunas 'Criptomoeda' e 'RetornoPercentual_MLP'.
    """
    if "RetornoPercentual_MLP" not in df_resultados.columns:
        print("[AVISO] Coluna 'RetornoPercentual_MLP' não encontrada. Gráfico de retorno padrão não será gerado.")
        return

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(df_resultados["Criptomoeda"], df_resultados["RetornoPercentual_MLP"], color="skyblue")
    plt.xlabel("Retorno Percentual (MLP)")
    plt.title("Retorno Percentual por Criptomoeda (Modelo MLP)")
    plt.tight_layout()
    plt.savefig("figures/retornos_criptos.png")
    plt.close()
    print("[OK] Gráfico salvo em figures/retornos_criptos.png")



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
