import os
import logging
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def executar_anova_por_grupo(caminho_csv: str, pasta_saida: str, coluna_agrupadora: str):
    """
    Executa uma ANOVA entre grupos formados com base em uma característica comum
    (como 'Média Retorno (%)').

    Args:
        caminho_csv (str): Caminho do arquivo CSV com os dados de criptomoedas.
        pasta_saida (str): Pasta onde os resultados da ANOVA serão salvos.
        coluna_agrupadora (str): Coluna usada para formar os grupos (ex: 'Média Retorno (%)').
    """
    try:
        df = pd.read_csv(caminho_csv)
        os.makedirs(pasta_saida, exist_ok=True)

        logging.info(f"[INÍCIO] ANOVA entre grupos com base em '{coluna_agrupadora}'.")

        # Converter para float se necessário
        df["Retorno"] = df["Média Retorno (%)"].astype(float)
        df[coluna_agrupadora] = df[coluna_agrupadora].astype(float)

        # Criar grupos (ex: baixo, médio, alto)
        quantis = df[coluna_agrupadora].quantile([0.33, 0.66])
        q1, q2 = quantis[0.33], quantis[0.66]

        def classificar(valor):
            if valor <= q1:
                return "Baixo"
            elif valor <= q2:
                return "Médio"
            else:
                return "Alto"

        df["Grupo"] = df[coluna_agrupadora].apply(classificar)
        df.to_csv(os.path.join(pasta_saida, "dados_com_grupo.csv"), index=False)

        # Médias por grupo
        medias = df.groupby("Grupo")["Retorno"].mean()
        logging.info("\n[Médias por Grupo]\n" + str(medias))
        medias.to_csv(os.path.join(pasta_saida, "medias_por_grupo.csv"))

        # ANOVA
        modelo = ols("Retorno ~ C(Grupo)", data=df).fit()
        anova = sm.stats.anova_lm(modelo, typ=2)
        logging.info("\n[ANOVA]\n" + str(anova))
        anova.to_csv(os.path.join(pasta_saida, "anova_resultados.csv"))

        # Teste de normalidade dos resíduos
        residuos = modelo.resid
        shapiro_test = shapiro(residuos)
        logging.info(
            f"[Shapiro-Wilk] Estatística = {shapiro_test.statistic:.4f}, "
            f"p-valor = {shapiro_test.pvalue:.4f}"
        )
        with open(os.path.join(pasta_saida, "shapiro_wilk.txt"), "w") as f:
            f.write(f"Estatística: {shapiro_test.statistic:.4f}\n")
            f.write(f"p-valor: {shapiro_test.pvalue:.4f}\n")

        # Teste de homogeneidade de variância (Levene)
        grupos = [g["Retorno"].values for _, g in df.groupby("Grupo")]
        levene_test = levene(*grupos)
        logging.info(
            f"[Levene] Estatística = {levene_test.statistic:.4f}, "
            f"p-valor = {levene_test.pvalue:.4f}"
        )
        with open(os.path.join(pasta_saida, "levene.txt"), "w") as f:
            f.write(f"Estatística: {levene_test.statistic:.4f}\n")
            f.write(f"p-valor: {levene_test.pvalue:.4f}\n")

        # Post hoc Tukey se ANOVA for significativa
        if anova["PR(>F)"][0] < 0.05:
            tukey = pairwise_tukeyhsd(endog=df["Retorno"], groups=df["Grupo"], alpha=0.05)
            logging.info("\n[Post Hoc - Tukey HSD]\n" + str(tukey.summary()))
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey_df.to_csv(os.path.join(pasta_saida, "tukey_posthoc_resultados.csv"), index=False)

        logging.info("[FIM] Análise ANOVA por grupo concluída com sucesso.")

    except Exception as e:
        logging.error(f"[ERRO] Durante execução da ANOVA por grupo: {e}")
