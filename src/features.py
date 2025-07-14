import pandas as pd
import numpy as np
import logging
from src.logging_config import configurar_logging

configurar_logging()


def adicionar_features_basicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas derivadas e estatísticas para melhorar a capacidade preditiva do modelo.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame com novas features.
    """
    df = df.copy()

    # Garantir os nomes padronizados
    df.rename(
        columns={
            "open": "Abertura",
            "high": "Alta",
            "low": "Baixa",
            "close": "Fechamento",
            "volume": "Volume",
        },
        inplace=True,
    )

    # Converter para numérico caso ainda não esteja
    for col in ["Abertura", "Alta", "Baixa", "Fechamento", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Retorno percentual (target implícito)
    df["Retorno"] = df["Fechamento"].pct_change()

    # Tendência de alta
    df["TendenciaAlta"] = (df["Fechamento"] > df["Abertura"]).astype(int)

    # Amplitude e variações intradiárias
    df["Amplitude"] = df["Alta"] - df["Baixa"]
    df["Variacao_Alta_Abertura"] = df["Alta"] - df["Abertura"]
    df["Variacao_Abertura_Baixa"] = df["Abertura"] - df["Baixa"]
    df["Razao_Alta_Baixa"] = df["Alta"] / df["Baixa"].replace(0, np.nan)
    df["Razao_Retorno_Desvio"] = df["Retorno"] / df["Retorno"].rolling(
        window=7
    ).std().replace(0, np.nan)

    # Médias móveis e desvios padrão
    df["MediaMovel_7d"] = df["Fechamento"].rolling(window=7).mean()
    df["MediaMovel_14d"] = df["Fechamento"].rolling(window=14).mean()
    df["MediaMovel_30d"] = df["Fechamento"].rolling(window=30).mean()
    df["DesvioPadrao_7d"] = df["Fechamento"].rolling(window=7).std()
    df["DesvioPadrao_14d"] = df["Fechamento"].rolling(window=14).std()

    # Volume médio
    if "Volume" in df.columns:
        df["MediaVolume_7d"] = df["Volume"].rolling(window=7).mean()
        df["MediaVolume_14d"] = df["Volume"].rolling(window=14).mean()

    # Lag features (últimos retornos)
    df["Retorno_Anterior_1d"] = df["Retorno"].shift(1)
    df["Retorno_Anterior_2d"] = df["Retorno"].shift(2)

    # Variáveis temporais
    if "Data" in df.columns:
        df["DiaSemana"] = df["Data"].dt.weekday
        df["Mes"] = df["Data"].dt.month
        df["FimDeSemana"] = (df["DiaSemana"] >= 5).astype(int)

    # Remover NaNs iniciais criados pelas médias móveis
    df = df.dropna().reset_index(drop=True)

    # Identificar features adicionadas (excluindo Data e Fechamento)
    features_adicionadas = list(set(df.columns) - set(["Data", "Fechamento"]))
    logging.info(f"[FEATURES] Features adicionadas com sucesso: {features_adicionadas}")
    logging.info(f"[FEATURES] Total de registros após limpeza: {df.shape[0]}")

    return df
