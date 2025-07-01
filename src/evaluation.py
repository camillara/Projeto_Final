import numpy as np
import pandas as pd
import logging
from typing import Tuple

def simular_estrategia_investimento(
    df: pd.DataFrame,
    previsoes: np.ndarray,
    threshold: float = 0.01,
    preco_col: str = "Fechamento",
    verbose: bool = True
) -> Tuple[float, pd.DataFrame]:
    """
    Simula uma estratégia de investimento com base nas previsões.

    Args:
        df (pd.DataFrame): DataFrame original com os dados reais.
        previsoes (np.ndarray): Previsões do modelo (valores futuros esperados).
        threshold (float): Limite mínimo de previsão de alta para comprar.
        preco_col (str): Nome da coluna de preço a ser usada.
        verbose (bool): Se True, imprime informações da simulação.

    Returns:
        Tuple[float, pd.DataFrame]: Lucro final e DataFrame com histórico da simulação.
    """
    capital = 10000  # capital inicial fictício
    capital_disponivel = capital
    lucro_total = 0
    historico = []

    for i in range(len(previsoes) - 1):
        preco_hoje = df[preco_col].iloc[i]
        preco_amanha = df[preco_col].iloc[i + 1]
        previsao = previsoes[i]

        variacao_prevista = (previsao - preco_hoje) / preco_hoje

        if variacao_prevista > threshold:
            quantidade_comprada = capital_disponivel / preco_hoje
            capital_final = quantidade_comprada * preco_amanha
            lucro = capital_final - capital_disponivel

            lucro_total += lucro
            historico.append({
                "Data": df['Data'].iloc[i],
                "PrecoHoje": preco_hoje,
                "PrecoAmanha": preco_amanha,
                "Previsao": previsao,
                "Lucro": round(lucro, 2)
            })

    df_resultado = pd.DataFrame(historico)

    if not df_resultado.empty:
        df_resultado["CapitalFinal"] = df_resultado["Lucro"].cumsum() + capital
        df_resultado["RetornoPercentual"] = (df_resultado["CapitalFinal"] - capital) / capital * 100
    else:
        df_resultado["CapitalFinal"] = [capital]
        df_resultado["RetornoPercentual"] = [0.0]

    if verbose:
        logging.info(f"[SIMULAÇÃO] Lucro final com capital inicial de R$ {capital:.2f}: R$ {lucro_total:.2f}")
        logging.info(f"[SIMULAÇÃO] Retorno percentual: {lucro_total / capital * 100:.2f}%")

    return lucro_total, df_resultado
