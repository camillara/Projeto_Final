import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from src.logging_config import configurar_logging


def calcular_erro_padrao(previsto_real_path: str = "results/previsto_real_por_modelo_por_cripto.csv",
                         output_path: str = "results/erro_padrao_modelos.csv") -> None:
    """
    Calcula o erro padrão (Root Mean Squared Error - RMSE) para cada combinação de criptomoeda e modelo.

    O cálculo é feito com base na diferença entre o valor previsto e o valor real de fechamento.

    Parâmetros:
    - previsto_real_path: Caminho para o arquivo CSV com os dados de previsão e valores reais.
    - output_path: Caminho para o arquivo CSV onde os erros padrão serão salvos.

    O arquivo de entrada deve conter as colunas: ['Criptomoeda', 'Modelo', 'Valor Real', 'Valor Previsto']
    """
    configurar_logging()

    logging.info("[INÍCIO] Cálculo do erro padrão (RMSE) para cada cripto e modelo...")

    # Verificar se o arquivo existe
    if not os.path.exists(previsto_real_path):
        logging.error(f"Arquivo não encontrado: {previsto_real_path}")
        return

    # Carregar dados
    df = pd.read_csv(previsto_real_path)

    # Verificar colunas obrigatórias
    colunas_esperadas = {"Criptomoeda", "Modelo", "Valor Real", "Valor Previsto"}
    if not colunas_esperadas.issubset(df.columns):
        logging.error(f"Arquivo não contém as colunas necessárias: {colunas_esperadas}")
        return

    # Agrupar por cripto e modelo e calcular RMSE
    resultados = []
    for (cripto, modelo), grupo in df.groupby(["Criptomoeda", "Modelo"]):
        rmse = np.sqrt(mean_squared_error(grupo["Valor Real"], grupo["Valor Previsto"]))
        resultados.append({
            "Criptomoeda": cripto,
            "Modelo": modelo,
            "RMSE": round(rmse, 6)
        })
        logging.info(f"[OK] {cripto} - {modelo} => RMSE = {rmse:.6f}")

    # Criar DataFrame e salvar
    df_rmse = pd.DataFrame(resultados)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_rmse.to_csv(output_path, index=False)
    logging.info(f"[SUCESSO] Erros padrão salvos em {output_path}")
