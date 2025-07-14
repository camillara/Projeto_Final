import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from src.logging_config import configurar_logging

# Inicializa logging
configurar_logging()
logging.info("[INÍCIO] Geração dos gráficos de lucro diário por modelo...")

# Lê o CSV
caminho_csv = "results/evolucao_lucro_diario.csv"
if not os.path.exists(caminho_csv):
    logging.error(f"Arquivo não encontrado: {caminho_csv}")
    exit(1)

df = pd.read_csv(caminho_csv)

# Converte a coluna Data, se necessário
try:
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
except Exception as e:
    logging.warning(f"Falha ao converter coluna Data: {e}")

# Cria pasta
pasta_saida = "figures/lucro_diario/por_modelo"
os.makedirs(pasta_saida, exist_ok=True)

# Itera por cripto e modelo
for cripto in df["Criptomoeda"].unique():
    for modelo in df["Modelo"].unique():
        df_plot = df[(df["Criptomoeda"] == cripto) & (df["Modelo"] == modelo)]

        if df_plot.empty:
            logging.warning(f"[AVISO] Nenhum dado para {cripto} - {modelo}")
            continue

        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df_plot, x="Data", y="CapitalFinal")
        plt.title(f"Evolução do Lucro - {cripto} - {modelo}")
        plt.ylabel("Capital Final (USD)")
        plt.xlabel("Data")
        plt.xticks(rotation=45)
        plt.tight_layout()

        nome_arquivo = f"{cripto}_{modelo}.png".replace("/", "_")
        caminho_figura = os.path.join(pasta_saida, nome_arquivo)
        plt.savefig(caminho_figura)
        plt.close()

        logging.info(f"[OK] Gráfico salvo: {caminho_figura}")
    
    # === Gera um único gráfico de lucro diário com todos os modelos por cripto ===
    pasta_todos_modelos = "figures/lucro_diario/todos_modelos"
    os.makedirs(pasta_todos_modelos, exist_ok=True)

    for cripto in df["Criptomoeda"].unique():
        df_cripto = df[df["Criptomoeda"] == cripto]

        if df_cripto.empty:
            logging.warning(f"[AVISO] Nenhum dado encontrado para {cripto}")
            continue

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_cripto, x="Data", y="CapitalFinal", hue="Modelo", marker=None)
        plt.title(f"Evolução do Lucro por Modelo - {cripto}")
        plt.ylabel("Capital Final (USD)")
        plt.xlabel("Data")
        plt.xticks(rotation=45)
        plt.legend(title="Modelo")
        plt.grid(True)
        plt.tight_layout()

        nome_arquivo = f"{cripto}_todos_modelos.png".replace("/", "_")
        caminho_figura = os.path.join(pasta_todos_modelos, nome_arquivo)
        plt.savefig(caminho_figura, dpi=150)
        plt.close()

        logging.info(f"[OK] Gráfico comparativo salvo: {caminho_figura}")

logging.info("[FIM] Geração dos gráficos por modelo concluída.")
