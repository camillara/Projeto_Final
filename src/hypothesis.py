from scipy import stats
from typing import Union
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def executar_teste_hipotese_retorno_diario_real(
    caminho_csv: str = "results/evolucao_lucro_diario.csv",
    retorno_esperado: float = 0.1,
    nivel_significancia: float = 0.05,
    salvar_csv: bool = True,
    caminho_saida: str = "results/teste_hipotese_retorno_diario.csv",
) -> pd.DataFrame:
    """
    Executa um teste t unilateral de hipótese para verificar se o retorno percentual diário médio
    de um modelo de previsão para criptomoedas é maior ou igual a um valor esperado definido pelo usuário.

    O teste é aplicado para cada combinação de Criptomoeda e Modelo presente no dataset.

    Hipóteses testadas:
        H₀ (hipótese nula):     μ ≥ retorno_esperado
        H₁ (hipótese alternativa): μ < retorno_esperado

    O teste utiliza os retornos diários calculados com base no campo 'CapitalFinal' do dataset.

    Parâmetros:
        caminho_csv (str): Caminho para o arquivo CSV contendo as colunas:
                           ['Data', 'Criptomoeda', 'Modelo', 'CapitalFinal'].
        retorno_esperado (float): Valor mínimo esperado para o retorno médio diário (em %).
                                  Exemplo: 0.1 significa 0.1% ao dia.
        nivel_significancia (float): Nível de significância do teste (padrão: 0.05).
        salvar_csv (bool): Se True, salva os resultados em um arquivo CSV.
        caminho_saida (str): Caminho de saída para o arquivo CSV de resultados, caso salvar_csv=True.

    Retorno:
        pd.DataFrame: DataFrame com os seguintes campos para cada par Criptomoeda/Modelo:
            - Criptomoeda: Nome da criptomoeda analisada
            - Modelo: Nome do modelo de previsão
            - Média Retorno (%): Média observada dos retornos diários
            - Retorno Esperado (%): Valor de referência definido pelo usuário
            - N dias: Tamanho da amostra usada no teste
            - Estatística t: Valor t calculado
            - p-valor: Valor-p do teste (unilateral)
            - Rejeita H₀ (médio ≥ x%): Indica se a hipótese nula foi rejeitada

    Observações:
        - A coluna 'CapitalFinal' deve representar o valor acumulado do capital ao final de cada dia.
        - O retorno diário é calculado como a variação percentual do capital de um dia para o seguinte.
        - Apenas séries com pelo menos 2 valores válidos são consideradas no teste.

    Exemplo:
        executar_teste_hipotese_retorno_diario_real(retorno_esperado=0.2)
    """

    logging.info("[INÍCIO] Carregando CSV com dados de capital final diário...")
    df = pd.read_csv(caminho_csv)
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    resultados = []

    logging.info(
        f"[INFO] Iniciando testes de hipótese para retorno esperado ≥ {retorno_esperado}%"
    )

    grupos = df.groupby(["Criptomoeda", "Modelo"])
    for (cripto, modelo), grupo in grupos:
        grupo = grupo.sort_values("Data")
        grupo["Retorno (%)"] = grupo["CapitalFinal"].pct_change() * 100
        amostra = grupo["Retorno (%)"].dropna()

        if len(amostra) < 2:
            logging.warning(
                f"[IGNORADO] {cripto} - {modelo}: amostra com menos de 2 valores."
            )
            continue

        t_stat, p_valor = stats.ttest_1samp(amostra, retorno_esperado)
        p_valor = stats.t.cdf(t_stat, df=len(amostra) - 1)

        rejeita_h0 = p_valor < nivel_significancia

        resultados.append(
            {
                "Criptomoeda": cripto,
                "Modelo": modelo,
                "Média Retorno (%)": round(amostra.mean(), 4),
                "Retorno Esperado (%)": retorno_esperado,
                "N dias": len(amostra),
                "Estatística t": round(t_stat, 4),
                "p-valor": round(p_valor, 5),
                "Rejeita H₀ (médio ≥ x%)": "Sim" if rejeita_h0 else "Não",
            }
        )

        logging.info(
            f"[OK] Testado {cripto} - {modelo}: p={p_valor:.5f} | Rejeita H₀: {'Sim' if rejeita_h0 else 'Não'}"
        )

    df_resultado = pd.DataFrame(resultados)

    if salvar_csv:
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        df_resultado.to_csv(caminho_saida, index=False)
        logging.info(f"[SUCESSO] Resultados salvos em: {caminho_saida}")

    logging.info("[FIM] Testes de hipótese concluídos.")
    return df_resultado
