import pandas as pd
import os
import logging
from src.logging_config import configurar_logging

# Criar diretório de logs (se ainda não existir)
os.makedirs("logs", exist_ok=True)

# Inicializar logging com seu configurador
configurar_logging("logs/gerar_correlacoes.log")

logging.info("Iniciando cálculo dos coeficientes de correlação...")

# Verificar existência do CSV
arquivo_csv = "results/previsto_real_por_modelo_por_cripto.csv"
if not os.path.exists(arquivo_csv):
    logging.error(f"Arquivo não encontrado: {arquivo_csv}")
    exit(1)

# Carregar o CSV
df = pd.read_csv(arquivo_csv)

# Armazenar os resultados
correlacoes = []

# Agrupar por Criptomoeda e Modelo
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

# Salvar os resultados
df_correlacoes = pd.DataFrame(correlacoes)
os.makedirs("results", exist_ok=True)
output_path = "results/coeficientes_correlacao_por_modelo.csv"
df_correlacoes.to_csv(output_path, index=False)

logging.info(f"[OK] Coeficientes de correlação salvos em {output_path}")
