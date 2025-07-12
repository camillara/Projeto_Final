
import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def executar_anova(caminho_csv: str, pasta_saida: str):
    try:
        df = pd.read_csv(caminho_csv)

        # ANOVA comparando as médias de retorno diário entre criptomoedas
        logging.info("[INÍCIO] Análise de variância entre criptomoedas.")

        # Apenas se necessário, converter porcentagem
        df["Retorno"] = df["Média Retorno (%)"].astype(float)

        modelo = ols("Retorno ~ C(Criptomoeda)", data=df).fit()
        anova = sm.stats.anova_lm(modelo, typ=2)
        logging.info("\n[ANOVA]\n" + str(anova))

        # Teste de normalidade dos resíduos
        residuos = modelo.resid
        shapiro_test = shapiro(residuos)
        logging.info(f"[Shapiro-Wilk] Estatística = {shapiro_test.statistic:.4f}, p-valor = {shapiro_test.pvalue:.4f}")

        # Teste de homogeneidade de variâncias de Levene
        grupos = [grupo["Retorno"].values for nome, grupo in df.groupby("Criptomoeda")]
        levene_test = levene(*grupos)
        logging.info(f"[Levene] Estatística = {levene_test.statistic:.4f}, p-valor = {levene_test.pvalue:.4f}")

        # Se ANOVA for significativa, aplica Tukey
        if anova["PR(>F)"][0] < 0.05:
            tukey = pairwise_tukeyhsd(endog=df["Retorno"], groups=df["Criptomoeda"], alpha=0.05)
            logging.info("\n[Post Hoc - Tukey HSD]\n" + str(tukey.summary()))

            # Salvar tabela de Tukey
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey_df.to_csv(os.path.join(pasta_saida, "tukey_posthoc_resultados.csv"), index=False)
            logging.info("[OK] Resultados do teste de Tukey salvos.")

    except Exception as e:
        logging.error(f"[ERRO] Durante execução da ANOVA: {e}")
