import os
import pandas as pd
from scripts.anova_por_grupo_caracteristica import executar_anova_por_grupo


def test_executar_anova_por_grupo(tmp_path):
    # Dados de teste
    dados = {
        "Criptomoeda": ["BTC", "ETH", "XRP", "BTC", "ETH", "XRP", "BTC", "ETH", "XRP"],
        "Média Retorno (%)": [0.5, 0.3, 0.2, 0.7, 0.4, 0.1, 0.6, 0.35, 0.15],
    }
    df = pd.DataFrame(dados)

    # Caminho para CSV temporário
    csv_path = tmp_path / "dados_teste.csv"
    df.to_csv(csv_path, index=False)

    # Pasta de saída
    pasta_saida = tmp_path / "resultados_anova"
    os.makedirs(pasta_saida, exist_ok=True)

    # Executar a função
    executar_anova_por_grupo(str(csv_path), str(pasta_saida), "Média Retorno (%)")

    # Verificar se os arquivos de resultado foram criados
    arquivos_esperados = [
        "dados_com_grupo.csv",
        "medias_por_grupo.csv",
        "anova_resultados.csv",
        "shapiro_wilk.txt",
        "levene.txt",
    ]

    for nome_arquivo in arquivos_esperados:
        caminho = os.path.join(pasta_saida, nome_arquivo)
        assert os.path.exists(caminho), f"Arquivo não encontrado: {nome_arquivo}"
