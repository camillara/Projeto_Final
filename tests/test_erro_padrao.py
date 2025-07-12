import os
import pandas as pd
import numpy as np
import tempfile
from scripts.erro_padrao import calcular_erro_padrao


def test_calcular_erro_padrao(tmp_path):
    # Criar DataFrame simulado com previsões e valores reais
    df_mock = pd.DataFrame({
        "Criptomoeda": ["BTC", "BTC", "ETH", "ETH"],
        "Modelo": ["MLP", "Linear", "MLP", "Linear"],
        "Valor Real": [100, 105, 200, 195],
        "Valor Previsto": [98, 107, 202, 190]
    })

    # Caminho para o CSV de entrada simulado
    csv_input = tmp_path / "previsto_real.csv"
    df_mock.to_csv(csv_input, index=False)

    # Caminho de saída esperado
    csv_output = tmp_path / "results" / "erro_padrao_modelos.csv"

    # Executar a função com os caminhos personalizados
    calcular_erro_padrao(previsto_real_path=str(csv_input), output_path=str(csv_output))

    # Verificar se o arquivo foi criado
    assert csv_output.exists(), "Arquivo de saída não foi criado"

    # Verificar conteúdo do CSV de saída
    df_saida = pd.read_csv(csv_output)
    assert not df_saida.empty, "Arquivo de saída está vazio"
    assert set(df_saida.columns) == {"Criptomoeda", "Modelo", "RMSE"}, "Colunas incorretas no arquivo de saída"

    # Verificar se RMSEs estão corretos
    for _, row in df_saida.iterrows():
        df_filtrado = df_mock[(df_mock["Criptomoeda"] == row["Criptomoeda"]) & (df_mock["Modelo"] == row["Modelo"])]
        esperado = np.sqrt(np.mean((df_filtrado["Valor Real"] - df_filtrado["Valor Previsto"]) ** 2))
        assert np.isclose(row["RMSE"], esperado, atol=1e-6), f"RMSE incorreto para {row['Criptomoeda']} - {row['Modelo']}"
