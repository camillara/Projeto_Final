import pandas as pd
import numpy as np
import os

# Carregar dados
df = pd.read_csv("results/previsto_real_por_modelo_por_cripto.csv")

# Pasta de saída
os.makedirs("results", exist_ok=True)

# Resultado final por cripto
dados_erro_padrao = []

for cripto in df["Criptomoeda"].unique():
    df_cripto = df[df["Criptomoeda"] == cripto]

    # Separar previsões por modelo
    previsoes_por_modelo = {}
    for modelo in df_cripto["Modelo"].unique():
        previsoes_por_modelo[modelo] = df_cripto[df_cripto["Modelo"] == modelo][
            "Valor Previsto"
        ].values

    # Calcular RMSEs (exceto MLP)
    y_real = df_cripto[df_cripto["Modelo"] == "MLP"][
        "Valor Real"
    ].values  # mesmo real para todos
    rmse_modelos = {
        modelo: np.sqrt(np.mean((y_real - preds) ** 2))
        for modelo, preds in previsoes_por_modelo.items()
        if modelo != "MLP"
    }

    if not rmse_modelos:
        print(f"[AVISO] Nenhum regressor encontrado para {cripto}")
        continue

    # Melhor regressor = menor RMSE (exceto MLP)
    melhor_modelo = min(rmse_modelos, key=rmse_modelos.get)

    # Cálculo do erro padrão entre MLP e o melhor regressor
    y_pred_mlp = previsoes_por_modelo["MLP"]
    y_pred_melhor = previsoes_por_modelo[melhor_modelo]

    diferencas = y_pred_mlp - y_pred_melhor
    erro_padrao = np.std(diferencas, ddof=1)  # ddof=1 para erro padrão da amostra

    dados_erro_padrao.append(
        {
            "Criptomoeda": cripto,
            "Melhor Regressor": melhor_modelo,
            "Erro Padrão (MLP vs Melhor)": erro_padrao,
        }
    )

# Salvar resultados
df_erro_padrao = pd.DataFrame(dados_erro_padrao)
df_erro_padrao.to_csv("results/erro_padrao_mlp_vs_melhor.csv", index=False)
print("[OK] Erro padrão salvo em results/erro_padrao_mlp_vs_melhor.csv")

import matplotlib.pyplot as plt
import seaborn as sns

# Pasta de figuras
pasta_figuras = "figures/erro_padrao"
os.makedirs(pasta_figuras, exist_ok=True)

# Ordenar pelo erro para visualização
df_erro_padrao.sort_values(
    by="Erro Padrão (MLP vs Melhor)", ascending=False, inplace=True
)

# Estilo
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Gráfico de barras
sns.barplot(
    x="Criptomoeda",
    y="Erro Padrão (MLP vs Melhor)",
    hue="Criptomoeda",
    data=df_erro_padrao,
    legend=False,
    palette="Blues_d",
)

plt.title("Erro Padrão entre MLP e Melhor Regressor por Criptomoeda")
plt.ylabel("Erro Padrão")
plt.xlabel("Criptomoeda")
plt.xticks(rotation=45)
plt.tight_layout()

# Caminho de saída
caminho_grafico = os.path.join(pasta_figuras, "erro_padrao_mlp_vs_melhor.png")
plt.savefig(caminho_grafico)
plt.close()

print(f"[OK] Gráfico salvo em: {caminho_grafico}")
