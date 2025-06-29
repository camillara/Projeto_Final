import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def adicionar_features_basicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas com features derivadas para previsão de preço de criptomoedas.

    Args:
        df (pd.DataFrame): DataFrame original com colunas ['Data', 'Fechamento', etc.]

    Returns:
        pd.DataFrame: DataFrame com novas features adicionadas.
    """
    df = df.copy()

    # Retorno diário (%)
    df['Retorno'] = df['Fechamento'].pct_change()

    # Média móvel (7 dias)
    df['MediaMovel_7d'] = df['Fechamento'].rolling(window=7).mean()

    # Desvio padrão (7 dias)
    df['DesvioPadrao_7d'] = df['Fechamento'].rolling(window=7).std()

    # Tendência (1 se Fechamento > Média Móvel, senão 0)
    df['TendenciaAlta'] = np.where(df['Fechamento'] > df['MediaMovel_7d'], 1, 0)

    # Eliminar primeiras linhas com NaN (rolling)
    df = df.dropna().reset_index(drop=True)

    logging.info(f"Features adicionadas: {df.columns.tolist()[-4:]}")
    return df
