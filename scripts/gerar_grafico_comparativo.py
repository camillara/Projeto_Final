import pandas as pd
from src.visualization import plot_grafico_comparativo_modelos

df_resultados = pd.read_csv("results/resultados_simulacoes.csv")


if "RetornoPercentual_Polinomial_2" in df_resultados.columns:
    df_resultados["RetornoPercentual_Poly_2"] = df_resultados["RetornoPercentual_Polinomial_2"]

plot_grafico_comparativo_modelos(df_resultados)
