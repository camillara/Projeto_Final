import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.logging_config import configurar_logging

# Inicializa o logging e cria diretórios necessários
os.makedirs("logs", exist_ok=True)
configurar_logging("logs/gerar_graficos_erro_padrao.log")
logging.info("Iniciando geração dos gráficos de erro padrão (RMSE) por criptomoeda.")

# Caminhos
ARQUIVO_CSV = "results/erro_padrao_modelos.csv"
DIRETORIO_SAIDA = "figures/erro_padrao"
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

# Verificação da existência do CSV
if not os.path.exists(ARQUIVO_CSV):
    logging.error(f"Arquivo não encontrado: {ARQUIVO_CSV}")
    exit(1)

# Leitura do arquivo
df = pd.read_csv(ARQUIVO_CSV)

# Validação das colunas necessárias
colunas_esperadas = {"Criptomoeda", "Modelo", "RMSE"}
if not colunas_esperadas.issubset(df.columns):
    logging.error(f"Colunas esperadas não encontradas. Esperado: {colunas_esperadas}")
    exit(1)

# Configuração visual
sns.set(style="whitegrid")

# Geração dos gráficos por criptomoeda
criptos = df["Criptomoeda"].unique()
for cripto in criptos:
    df_cripto = df[df["Criptomoeda"] == cripto].copy()
    df_cripto.sort_values(by="RMSE", ascending=True, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="RMSE", y="Modelo", data=df_cripto, hue="Modelo", palette="crest", legend=False)
    plt.title(f"Erro Padrão (RMSE) por Modelo - {cripto}")
    plt.xlabel("Erro Padrão (RMSE)")
    plt.ylabel("Modelo")
    plt.tight_layout()

    caminho_saida = os.path.join(DIRETORIO_SAIDA, f"grafico_erro_padrao_{cripto}.png")
    plt.savefig(caminho_saida)
    plt.close()
    logging.info(f"[OK] Gráfico salvo: {caminho_saida}")
