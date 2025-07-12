import os
import pandas as pd
from scripts.anova_retorno_criptos import executar_anova

def test_executar_anova(tmp_path):
    # Criar dados de exemplo com retorno médio diário por criptomoeda
    dados = {
        "Criptomoeda": ["BTC", "ETH", "XRP", "BTC", "ETH", "XRP", "BTC", "ETH", "XRP"],
        "Média Retorno (%)": [0.5, 0.3, 0.2, 0.7, 0.4, 0.1, 0.6, 0.35, 0.15]
    }
    df = pd.DataFrame(dados)

    # Caminho para CSV temporário
    csv_path = tmp_path / "dados_teste.csv"
    df.to_csv(csv_path, index=False)

    # Pasta de saída
    pasta_saida = tmp_path / "saida_anova"
    os.makedirs(pasta_saida, exist_ok=True)

    # Executar a função
    executar_anova(str(csv_path), str(pasta_saida))

    # Verificar se arquivos de saída foram gerados
    arquivos_esperados = [
        "medias_por_criptomoeda.csv",
        "anova_resultados.csv",
        "shapiro_wilk.txt",
        "levene.txt"
    ]

    for nome_arquivo in arquivos_esperados:
        caminho = os.path.join(pasta_saida, nome_arquivo)
        assert os.path.exists(caminho), f"Arquivo não encontrado: {nome_arquivo}"

