import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from src.utils import preprocessar_dados

# Diretórios
DIRETORIO_DADOS = "data"
DIRETORIO_MODELOS = "modelos"
DIRETORIO_SAIDA = "figures"

# Lista de criptomoedas
criptos = [
    "DASHUSDT", "XRPUSDT", "XMRUSDT", "ETHUSDT", "LTCUSDT",
    "ZRXUSDT", "BTCUSDT", "BATUSDT", "BCHUSDT", "ETCUSDT"
]

# Sufixos dos modelos
modelos = {
    "MLP": "_mlp.joblib",
    "Linear": "_linear.joblib"
}
for grau in range(2, 11):
    modelos[f"Polinomial_{grau}"] = f"_polinomial_grau{grau}.joblib"

# Geração dos gráficos
for cripto in criptos:
    caminho_csv = os.path.join(DIRETORIO_DADOS, f"Poloniex_{cripto}_d.csv")
    if not os.path.exists(caminho_csv):
        print(f"[AVISO] Arquivo CSV não encontrado: {caminho_csv}")
        continue

    df = pd.read_csv(caminho_csv, skiprows=1)

    try:
        df = preprocessar_dados(df)
    except Exception as e:
        print(f"[ERRO] Falha ao preprocessar {cripto}: {e}")
        continue

    y_real = df["Fechamento"]
    X = df.drop(columns=["Fechamento"], errors="ignore")

    plt.figure(figsize=(10, 6))

    for nome_modelo, sufixo in modelos.items():
        caminho_modelo = os.path.join(DIRETORIO_MODELOS, f"{cripto}{sufixo}")
        if not os.path.exists(caminho_modelo):
            print(f"[AVISO] Modelo não encontrado: {caminho_modelo}")
            continue

        try:
            modelo = joblib.load(caminho_modelo)
            y_pred = modelo.predict(X)
            plt.scatter(y_real, y_pred, alpha=0.4, label=nome_modelo, s=15)
        except Exception as e:
            print(f"[ERRO] {nome_modelo} em {cripto}: {e}")

    min_val = min(y_real.min(), y_real.max())
    max_val = max(y_real.max(), y_real.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')

    plt.xlabel("Valor Real (Fechamento)")
    plt.ylabel("Valor Previsto")
    plt.title(f"Dispersão - Modelos para {cripto}")
    plt.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    caminho_saida = os.path.join(DIRETORIO_SAIDA, f"{cripto}_dispersao_modelos.png")
    plt.savefig(caminho_saida)
    plt.close()

    print(f"[OK] Gráfico salvo: {caminho_saida}")