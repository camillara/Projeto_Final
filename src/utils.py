import os
import matplotlib.pyplot as plt
import joblib
import logging
from typing import Any
import pandas as pd

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

# === PREPROCESSAMENTO DE DADOS ===
def preprocessar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função de pré-processamento padrão para os datasets de criptomoedas.

    - Converte colunas para float se necessário.
    - Renomeia 'close' para 'Fechamento' e calcula o 'Retorno'.
    - Remove colunas desnecessárias como 'Unnamed: 0', 'unix', 'symbol', etc.
    - Padroniza nomes de colunas para consistência com os modelos treinados.

    Args:
        df (pd.DataFrame): DataFrame original carregado do CSV.

    Returns:
        pd.DataFrame: DataFrame limpo e pronto para uso.
    """
    df = df.copy()

    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        df = df.sort_values("Data").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Renomeia 'close' se necessário
    if "close" in df.columns and "Fechamento" not in df.columns:
        df.rename(columns={"close": "Fechamento"}, inplace=True)

    if "Fechamento" in df.columns:
        df["Retorno"] = df["Fechamento"].pct_change()
        df["MediaMovel_7d"] = df["Fechamento"].rolling(window=7).mean()
        df["DesvioPadrao_7d"] = df["Fechamento"].rolling(window=7).std()

    if "Fechamento" in df.columns and "Abertura" in df.columns:
        df["TendenciaAlta"] = (df["Fechamento"] > df["Abertura"]).astype(int)

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    colunas_validas = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if "Fechamento" in df.columns:
        colunas_validas.append("Fechamento")
    colunas_validas = list(set(colunas_validas))  # remove duplicatas

    import logging
    logging.info(f"[FEATURES] Features adicionadas com sucesso: {list(set(df.columns) - set(['Data', 'Fechamento']))}")
    logging.info(f"[FEATURES] Total de registros após limpeza: {len(df)}")

    return df[colunas_validas + (["Data"] if "Data" in df.columns else [])]