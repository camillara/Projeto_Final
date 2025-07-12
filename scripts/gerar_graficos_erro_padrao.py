import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.logging_config import configurar_logging

def gerar_graficos_erro_padrao(csv_path: str = "results/erro_padrao_modelos.csv", pasta_saida: str = "figures/erro_padrao") -> None:
    """
    Gera gráficos de barras com o erro padrão (RMSE) por modelo para cada criptomoeda.

    Esta função lê um arquivo CSV contendo os valores de RMSE de diferentes modelos
    aplicados a várias criptomoedas, e cria gráficos salvos em arquivos PNG que
    comparam visualmente o desempenho dos modelos para cada ativo.

    Parâmetros:
    ----------
    csv_path : str, opcional
        Caminho para o arquivo CSV contendo os dados de erro padrão.
        Deve conter as colunas: 'Criptomoeda', 'Modelo' e 'RMSE'.
        Padrão: "results/erro_padrao_modelos.csv".

    pasta_saida : str, opcional
        Caminho para o diretório onde os gráficos serão salvos.
        O diretório será criado se não existir.
        Padrão: "figures/erro_padrao".

    Logs:
    ----
    Cria um arquivo de log em "logs/gerar_graficos_erro_padrao.log" contendo informações sobre a execução.

    Erros Tratados:
    --------------
    - Arquivo CSV inexistente.
    - Colunas ausentes no CSV.

    Retorno:
    -------
    None
    """
    # Inicializa o logging e cria diretórios necessários
    os.makedirs("logs", exist_ok=True)
    configurar_logging("logs/gerar_graficos_erro_padrao.log")
    logging.info("Iniciando geração dos gráficos de erro padrão (RMSE) por criptomoeda.")

    # Cria pasta de saída se necessário
    os.makedirs(pasta_saida, exist_ok=True)

    # Verificação da existência do CSV
    if not os.path.exists(csv_path):
        logging.error(f"Arquivo não encontrado: {csv_path}")
        return

    # Leitura do arquivo
    df = pd.read_csv(csv_path)

    # Validação das colunas necessárias
    colunas_esperadas = {"Criptomoeda", "Modelo", "RMSE"}
    if not colunas_esperadas.issubset(df.columns):
        logging.error(f"Colunas esperadas não encontradas. Esperado: {colunas_esperadas}")
        return

    # Configuração visual
    sns.set(style="whitegrid")

    # Geração dos gráficos por criptomoeda
    criptos = df["Criptomoeda"].unique()
    for cripto in criptos:
        df_cripto = df[df["Criptomoeda"] == cripto].copy()
        df_cripto.sort_values(by="RMSE", ascending=True, inplace=True)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="RMSE", y="Modelo", data=df_cripto, hue="Modelo", palette="crest", legend=False)
        plt.title(f"Erro Padrão (RMSE) por Modelo - {cripto}")
        plt.xlabel("Erro Padrão (RMSE)")
        plt.ylabel("Modelo")
        plt.tight_layout()

        # Adiciona os valores ao lado ou dentro da barra, conforme o valor
        max_rmse = df_cripto["RMSE"].max()
        threshold = max_rmse * 0.15  # define ponto de corte para mudar a posição do texto

        for i, (rmse, modelo) in enumerate(zip(df_cripto["RMSE"], df_cripto["Modelo"])):
            if rmse < threshold:
                # texto à direita da barra
                ax.text(rmse + max_rmse * 0.01, i, f"{rmse:.4f}", va='center', ha='left', fontsize=9, color='black')
            else:
                # texto dentro da barra (alinhado à direita)
                ax.text(rmse - max_rmse * 0.01, i, f"{rmse:.4f}", va='center', ha='right', fontsize=9, color='white')

        caminho_saida = os.path.join(pasta_saida, f"grafico_erro_padrao_{cripto}.png")
        plt.savefig(caminho_saida)
        plt.close()
        logging.info(f"[OK] Gráfico salvo: {caminho_saida}")
        
        # === 1. Gráfico Comparativo Geral ===
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=df, x="Modelo", y="RMSE", hue="Criptomoeda", errorbar=None)
        plt.title("Comparativo Geral de RMSE por Modelo e Criptomoeda")
        plt.ylabel("Erro Padrão (RMSE)")
        plt.xlabel("Modelo")
        plt.xticks(rotation=45)
        plt.tight_layout()

        comparativo_path = os.path.join(pasta_saida, "comparativo_geral_modelos.png")
        plt.savefig(comparativo_path)
        plt.close()
        logging.info(f"[OK] Gráfico comparativo geral salvo: {comparativo_path}")

        # === 2. Ranking de RMSE Médio por Modelo ===
        ranking = (
            df.groupby("Modelo")["RMSE"]
            .mean()
            .sort_values()
            .reset_index()
            .rename(columns={"RMSE": "RMSE Médio"})
        )

        pasta_results = "results"
        ranking_path = os.path.join(pasta_results, "ranking_rmse_modelos.csv")
        ranking.to_csv(ranking_path, index=False)
        logging.info(f"[OK] Ranking dos modelos salvo: {ranking_path}")

        # Gráfico do ranking dos modelos com menor RMSE médio
        plt.figure(figsize=(10, 6))
        ranking["Hue"] = ranking["Modelo"]  # dummy hue para ativar o uso de palette corretamente
        ax = sns.barplot(data=ranking, x="RMSE Médio", y="Modelo", hue="Hue", palette="viridis", legend=False)
        if ax.legend_ is not None:
            ax.legend_.remove()
    

        # Adiciona os valores ao lado de cada barra
        for i, v in enumerate(ranking["RMSE Médio"]):
            ax.text(v + 5, i, f"{v:.2f}", va="center", fontsize=9)

        plt.title("Ranking dos Modelos por Menor RMSE Médio")
        plt.xlabel("RMSE Médio")
        plt.ylabel("Modelo")
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "ranking_rmse_modelos.png"))
        plt.close()
