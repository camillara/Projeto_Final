import pandas as pd
import numpy as np
import logging
from typing import List
from src.logging_config import configurar_logging

configurar_logging()

def adicionar_features_basicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas com features derivadas para previsão de preço de criptomoedas.

    Features adicionadas:
        - Retorno: Variação percentual diária do preço de fechamento.
        - MediaMovel_7d: Média móvel de 7 dias do fechamento.
        - DesvioPadrao_7d: Desvio padrão da média móvel de 7 dias.
        - TendenciaAlta: Indicador binário se o fechamento está acima da média móvel.

    Args:
        df (pd.DataFrame): DataFrame contendo pelo menos a coluna 'Fechamento'.

    Returns:
        pd.DataFrame: DataFrame com as novas features adicionadas e linhas iniciais com NaN removidas.
    """
    df = df.copy()

    # Calcular retorno diário
    df['Retorno'] = df['Fechamento'].pct_change()

    # Calcular média móvel de 7 dias
    df['MediaMovel_7d'] = df['Fechamento'].rolling(window=7).mean()

    # Calcular desvio padrão de 7 dias
    df['DesvioPadrao_7d'] = df['Fechamento'].rolling(window=7).std()

    # Indicador de tendência de alta
    df['TendenciaAlta'] = np.where(df['Fechamento'] > df['MediaMovel_7d'], 1, 0)

    # Remover linhas com NaN geradas pelas operações de rolling e pct_change
    df = df.dropna().reset_index(drop=True)

    logging.info(f"[FEATURES] Features adicionadas com sucesso: {df.columns.tolist()[-4:]}")
    logging.info(f"[FEATURES] Total de registros após limpeza: {df.shape[0]}")
    
    return df
