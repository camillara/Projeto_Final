import os
import matplotlib.pyplot as plt

def plot_grafico_retorno(df_resultados):
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(df_resultados["Criptomoeda"], df_resultados["RetornoPercentual"], color="skyblue")
    plt.xlabel("Retorno Percentual")
    plt.title("Retorno por Criptomoeda (Simulação MLP)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("figures/retornos_criptos.png")
    plt.close()
