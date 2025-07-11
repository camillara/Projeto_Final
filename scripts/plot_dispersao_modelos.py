import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.logging_config import configurar_logging

# Inicializar logging
os.makedirs("logs", exist_ok=True)
configurar_logging("logs/plot_dispersao_modelos.log")
logging.info("Iniciando geração dos gráficos de dispersão e cálculo de correlação.")

# Caminhos
ARQUIVO_CSV = "previsto_real_por_modelo_por_cripto.csv"
DIRETORIO_SAIDA = "figures/dispersao"
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

# Verificar existência do CSV
if not os.path.exists(ARQUIVO_CSV):
    logging.error(f"Arquivo não encontrado: {ARQUIVO_CSV}")
    exit(1)

# Carregar os dados
df = pd.read_csv(ARQUIVO_CSV)

# Obter todas as criptos
criptos = df["Criptomoeda"].unique()
sns.set(style="whitegrid")

# Gerar gráficos de dispersão
for cripto in criptos:
    logging.info(f"Gerando gráfico de dispersão para {cripto}")
    plt.figure(figsize=(10, 6))
    df_cripto = df[df["Criptomoeda"] == cripto]

    for modelo in df_cripto["Modelo"].unique():
        df_modelo = df_cripto[df_cripto["Modelo"] == modelo]
        plt.scatter(df_modelo["Valor Real"], df_modelo["Valor Previsto"], 
                    label=modelo, alpha=0.5, s=15)

    min_val = min(df_cripto["Valor Real"].min(), df_cripto["Valor Previsto"].min())
    max_val = max(df_cripto["Valor Real"].max(), df_cripto["Valor Previsto"].max())

    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
    plt.xlabel("Valor Real (Fechamento)")
    plt.ylabel("Valor Previsto")
    plt.title(f"Dispersão - Modelos para {cripto}")
    plt.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    caminho_saida = os.path.join(DIRETORIO_SAIDA, f"{cripto}_dispersao_modelos.png")
    plt.savefig(caminho_saida)
    plt.close()
    logging.info(f"[OK] Gráfico salvo: {caminho_saida}")

# Cálculo dos coeficientes de correlação (item b)
logging.info("Calculando coeficientes de correlação entre valor real e previsto...")

correlacoes = []
for (cripto, modelo), grupo in df.groupby(["Criptomoeda", "Modelo"]):
    if len(grupo) > 1:
        correlacao = grupo[["Valor Real", "Valor Previsto"]].corr().iloc[0, 1]
    else:
        correlacao = None
        logging.warning(f"Grupo com poucos dados para correlação: {cripto} - {modelo}")

    correlacoes.append({
        "Criptomoeda": cripto,
        "Modelo": modelo,
        "Correlacao": correlacao
    })

df_correlacoes = pd.DataFrame(correlacoes)
os.makedirs("results", exist_ok=True)
output_correlacao = "results/coeficientes_correlacao_por_modelo.csv"
df_correlacoes.to_csv(output_correlacao, index=False)
logging.info(f"[OK] Coeficientes de correlação salvos em {output_correlacao}")
