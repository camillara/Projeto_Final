import os
import logging
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def executar_anova(caminho_csv: str, pasta_saida: str) -> None:
    """
    Executa a análise de variância (ANOVA) para comparar os retornos médios diários entre criptomoedas.

    Esta função realiza os seguintes passos:
    1. Carrega os dados de retorno médio diário a partir de um arquivo CSV.
    2. Calcula as médias por criptomoeda e salva em arquivo.
    3. Realiza a ANOVA para verificar diferenças significativas entre as médias.
    4. Executa testes de suposições: Shapiro-Wilk (normalidade dos resíduos) e Levene (homogeneidade das variâncias).
    5. Caso a ANOVA seja significativa, executa o teste post-hoc de Tukey HSD.
    6. Salva todos os resultados em arquivos dentro da pasta especificada.

    Parâmetros:
    ----------
    caminho_csv : str
        Caminho para o arquivo CSV contendo as colunas 'Criptomoeda' e 'Média Retorno (%)'.

    pasta_saida : str
        Caminho da pasta onde os arquivos de resultados (CSV e TXT) serão salvos.

    Retorno:
    -------
    None
    """
    try:
        df = pd.read_csv(caminho_csv)

        # Garante que a pasta de saída exista
        os.makedirs(pasta_saida, exist_ok=True)

        logging.info("[INÍCIO] Análise de variância entre criptomoedas.")
        df["Retorno"] = df["Média Retorno (%)"].astype(float)

        # Médias por criptomoeda
        medias = df.groupby("Criptomoeda")["Retorno"].mean()
        logging.info("\n[Médias por Criptomoeda]\n" + str(medias))
        medias.to_csv(os.path.join(pasta_saida, "medias_por_criptomoeda.csv"))

        # ANOVA
        modelo = ols("Retorno ~ C(Criptomoeda)", data=df).fit()
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

        # Teste de homogeneidade de variâncias (Levene)
        grupos = [grupo["Retorno"].values for _, grupo in df.groupby("Criptomoeda")]
        levene_test = levene(*grupos)
        logging.info(
            f"[Levene] Estatística = {levene_test.statistic:.4f}, "
            f"p-valor = {levene_test.pvalue:.4f}"
        )
        with open(os.path.join(pasta_saida, "levene.txt"), "w") as f:
            f.write(f"Estatística: {levene_test.statistic:.4f}\n")
            f.write(f"p-valor: {levene_test.pvalue:.4f}\n")

        # Teste de Tukey se ANOVA for significativa
        if anova["PR(>F)"].iloc[0] < 0.05:
            tukey = pairwise_tukeyhsd(endog=df["Retorno"], groups=df["Criptomoeda"], alpha=0.05)
            logging.info("\n[Post Hoc - Tukey HSD]\n" + str(tukey.summary()))
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey_df.to_csv(os.path.join(pasta_saida, "tukey_posthoc_resultados.csv"), index=False)

        logging.info("[FIM] Análise ANOVA concluída com sucesso.")

    except Exception as e:
        logging.error(f"[ERRO] Durante execução da ANOVA: {e}")
