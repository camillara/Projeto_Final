import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminhos
ARQUIVO_CSV = "previsto_real_por_modelo_por_cripto.csv"
DIRETORIO_SAIDA = "figures"

# Criar pasta de saída, se necessário
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

# Carregar os dados
df = pd.read_csv(ARQUIVO_CSV)

# Obter todas as criptos
criptos = df["Criptomoeda"].unique()

# Configurar estilo do seaborn
sns.set(style="whitegrid")

for cripto in criptos:
    plt.figure(figsize=(10, 6))
    df_cripto = df[df["Criptomoeda"] == cripto]

    # Desenhar gráfico de dispersão por modelo
    for modelo in df_cripto["Modelo"].unique():
        df_modelo = df_cripto[df_cripto["Modelo"] == modelo]
        plt.scatter(df_modelo["Valor Real"], df_modelo["Valor Previsto"], 
                    label=modelo, alpha=0.5, s=15)

    min_val = min(df_cripto["Valor Real"].min(), df_cripto["Valor Previsto"].min())
    max_val = max(df_cripto["Valor Real"].max(), df_cripto["Valor Previsto"].max())

    # Linha de referência y = x
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')

    plt.xlabel("Valor Real (Fechamento)")
    plt.ylabel("Valor Previsto")
    plt.title(f"Dispersão - Modelos para {cripto}")
    plt.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    caminho_saida = os.path.join(DIRETORIO_SAIDA, f"{cripto}_dispersao_modelos.png")
    plt.savefig(caminho_saida)
    plt.close()

    print(f"[OK] Gráfico salvo: {caminho_saida}")
