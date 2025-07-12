import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def plotar_graficos_teste_hipotese(
    caminho_csv: str = "results/teste_hipotese_retorno_diario.csv",
    pasta_saida: str = "figures/hipotese"
):
    """
    Gera três gráficos para ilustrar os resultados do teste de hipótese de retorno médio:
    1. Gráfico de barras dos valores-p por criptomoeda e modelo
    2. Gráfico de barras das médias de retorno observadas vs esperado
    3. Heatmap indicando rejeição ou não da hipótese nula

    Args:
        caminho_csv (str): Caminho para o CSV com os resultados dos testes de hipótese.
        pasta_saida (str): Diretório onde os gráficos serão salvos.
    """
    logging.info("[INÍCIO] Carregando dados do teste de hipótese...")
    
    try:
        df = pd.read_csv(caminho_csv)
    except FileNotFoundError:
        logging.error(f"[ERRO] Arquivo não encontrado: {caminho_csv}")
        return

    if df.empty:
        logging.warning("[AVISO] O arquivo está vazio. Nenhum gráfico será gerado.")
        return

    os.makedirs(pasta_saida, exist_ok=True)

    # === 1. Gráfico de barras do p-valor ===
    try:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="Criptomoeda", y="p-valor", hue="Modelo")
        plt.axhline(0.05, color="red", linestyle="--", label="Nível de significância (0.05)")
        plt.title("Valores-p do Teste de Hipótese por Criptomoeda e Modelo")
        plt.ylabel("p-valor")
        plt.xlabel("Criptomoeda")

        # Legenda fora do gráfico (na lateral direita)
        plt.legend(
            bbox_to_anchor=(1.02, 1), 
            loc='upper left', 
            borderaxespad=0
        )

        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Dá espaço para a legenda na direita

        caminho_fig1 = os.path.join(pasta_saida, "p_valores_modelos.png")
        plt.savefig(caminho_fig1, bbox_inches="tight")
        plt.close()
        logging.info(f"[OK] Gráfico 1 salvo em: {caminho_fig1}")
    except Exception as e:
        logging.error(f"[ERRO] Ao gerar gráfico de p-valores: {e}")


    # === 2. Gráfico de barras das médias de retorno ===
    try:
        criptos = df["Criptomoeda"].unique()
        retorno_esperado = df["Retorno Esperado (%)"].iloc[0]  # Assumimos que é fixo para todos

        for cripto in criptos:
            df_cripto = df[df["Criptomoeda"] == cripto]

            plt.figure(figsize=(10, 5))
            sns.barplot(data=df_cripto, x="Modelo", y="Média Retorno (%)")
            plt.axhline(retorno_esperado, color="green", linestyle="--",
                        label=f"Retorno Esperado ({retorno_esperado:.1f}%)")
            plt.title(f"Média de Retorno Diário - {cripto}")
            plt.ylabel("Média de Retorno (%)")
            plt.xlabel("Modelo")
            plt.xticks(rotation=90)

            # Legenda com % e fora do gráfico
            plt.legend(
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0
            )

            plt.tight_layout(rect=[0, 0, 0.85, 1])
            caminho_fig = os.path.join(pasta_saida, f"media_retornos_{cripto}.png")
            plt.savefig(caminho_fig, bbox_inches="tight")
            plt.close()
            logging.info(f"[OK] Gráfico salvo para {cripto}: {caminho_fig}")

    except Exception as e:
        logging.error(f"[ERRO] Ao gerar gráfico de médias de retorno por cripto: {e}")


    # === 3. Heatmap de rejeição da hipótese ===
    try:
        df["Rejeita"] = df["Rejeita H₀ (médio ≥ x%)"].map({"Sim": 1, "Não": 0})
        heatmap_data = df.pivot(index="Modelo", columns="Criptomoeda", values="Rejeita")

        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(heatmap_data, annot=True, cmap="Reds", cbar=False, fmt="d")
        plt.title("Heatmap: Rejeição da Hipótese Nula (H₀)")

        # Rotacionar nomes das criptos
        plt.xticks(rotation=90)

        # Adiciona legenda lateral
        legenda_texto = "Legenda:\n1 = Rejeita H₀\n0 = Não Rejeita H₀"
        plt.text(
            heatmap_data.shape[1] + 0.6,
            -0.5,
            legenda_texto,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        caminho_fig3 = os.path.join(pasta_saida, "heatmap_rejeicao.png")
        plt.savefig(caminho_fig3, bbox_inches="tight")
        plt.close()
        logging.info(f"[OK] Gráfico 3 salvo em: {caminho_fig3}")

    except Exception as e:
        logging.error(f"[ERRO] Ao gerar heatmap de rejeição: {e}")


