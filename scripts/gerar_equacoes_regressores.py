import os
import gc
import joblib
import logging
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from src.logging_config import configurar_logging

# Inicializar logging
os.makedirs("logs", exist_ok=True)
configurar_logging("logs/gerar_equacoes_regressores.log")
logging.info("Iniciando geração das equações dos regressores...")

# Caminhos
DIRETORIO_MODELOS = "modelos"
DIRETORIO_DADOS = "data"
DIRETORIO_SAIDA = "results"
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

# Lista de criptos
criptos = [
    "DASHUSDT", "XRPUSDT", "XMRUSDT", "ETHUSDT", "LTCUSDT",
    "ZRXUSDT", "BTCUSDT", "BATUSDT", "BCHUSDT", "ETCUSDT"
]

# Log de arquivos de modelos disponíveis
if os.path.exists(DIRETORIO_MODELOS):
    logging.info("Modelos disponíveis:")
    for f in sorted(os.listdir(DIRETORIO_MODELOS)):
        logging.info(f" - {f}")

equacoes = []

for cripto in criptos:
    logging.info(f"\n[CRYPTO] Processando {cripto}...")

    # Carregar dados
    caminho_csv = os.path.join(DIRETORIO_DADOS, f"Poloniex_{cripto}_d.csv")
    if not os.path.exists(caminho_csv):
        logging.warning(f"Arquivo CSV não encontrado: {caminho_csv}")
        continue

    try:
        df = pd.read_csv(caminho_csv, skiprows=1)
        from src.utils import preprocessar_dados
        df = preprocessar_dados(df)
        X = df.drop(columns=["Fechamento", "Data"], errors="ignore")
        feature_names = X.columns.tolist()
    except Exception as e:
        logging.error(f"Erro ao processar dados de {cripto}: {e}")
        continue

    # Modelos: Linear e Polinomiais grau 2 até 5
    for grau in range(1, 6):  # Grau 1 = Linear
        nome_modelo = f"{cripto}_linear.joblib" if grau == 1 else f"{cripto}_polinomial_grau{grau}.joblib"
        modelo_path = os.path.join(DIRETORIO_MODELOS, nome_modelo)

        if not os.path.exists(modelo_path):
            logging.warning(f"Modelo não encontrado: {modelo_path}")
            continue

        try:
            modelo = joblib.load(modelo_path)

            if grau == 1:
                coef = modelo.coef_
                intercept = modelo.intercept_
                nomes_features = feature_names
            else:
                poly = PolynomialFeatures(degree=grau)
                poly.fit(X)  # Apenas para pegar nomes
                nomes_features = poly.get_feature_names_out(feature_names)
                coef = modelo.named_steps['linearregression'].coef_
                intercept = modelo.named_steps['linearregression'].intercept_

            termos = [f"{coef[i]:.4f}*{nomes_features[i]}" for i in range(len(coef))]
            equacao = " + ".join(termos)
            equacao = f"y = {intercept:.4f} + " + equacao

            equacoes.append({
                "Criptomoeda": cripto,
                "Modelo": "Linear" if grau == 1 else f"Polinomial_{grau}",
                "Equacao": equacao
            })

            logging.info(f"✔️ Equação gerada: {cripto} - Grau {grau}")
        except Exception as e:
            logging.error(f"Erro ao gerar equação para {cripto} - Grau {grau}: {e}")
        finally:
            del modelo
            gc.collect()

# Salvar no CSV
df_equacoes = pd.DataFrame(equacoes)
caminho_saida = os.path.join(DIRETORIO_SAIDA, "equacoes_regressores.csv")
df_equacoes.to_csv(caminho_saida, index=False)
logging.info(f"\n[OK] Equações salvas em {caminho_saida}")
