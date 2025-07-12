import os
import pandas as pd
import numpy as np
import subprocess
import sys


def test_gerar_correlacoes(tmp_path):
    # Criar dados simulados
    df_simulado = pd.DataFrame({
        "Criptomoeda": ["BTC", "BTC", "ETH", "ETH"],
        "Modelo": ["MLP", "MLP", "Linear", "Linear"],
        "Valor Real": [100, 105, 200, 190],
        "Valor Previsto": [102, 103, 198, 192]
    })

    # Caminhos temporários
    arquivo_csv = tmp_path / "previsto_real_por_modelo_por_cripto.csv"
    df_simulado.to_csv(arquivo_csv, index=False)

    pasta_scripts = tmp_path / "scripts"
    pasta_scripts.mkdir()
    pasta_logs = tmp_path / "logs"
    pasta_logs.mkdir()
    pasta_results = tmp_path / "results"
    pasta_results.mkdir()

    # Copiar e adaptar o script
    script_origem = "scripts/gerar_correlacoes.py"
    script_destino = pasta_scripts / "gerar_correlacoes.py"
    with open(script_origem, "r") as f:
        conteudo = f.read().replace("previsto_real_por_modelo_por_cripto.csv", str(arquivo_csv))
    with open(script_destino, "w") as f:
        f.write(conteudo)

    # Executar o script com PYTHONPATH configurado
    resultado = subprocess.run(
        ["python3", str(script_destino)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(os.getcwd())}  # adiciona o diretório raiz do projeto
    )

    assert resultado.returncode == 0, f"Erro ao executar script: {resultado.stderr}"

    # Validar arquivo de saída
    output_path = tmp_path / "results" / "coeficientes_correlacao_por_modelo.csv"
    assert output_path.exists(), "Arquivo de saída de correlação não foi criado"

    df_correlacao = pd.read_csv(output_path)
    assert not df_correlacao.empty, "Arquivo de correlação está vazio"
    assert set(df_correlacao.columns) == {"Criptomoeda", "Modelo", "Correlacao"}

    # Verificar valor da correlação
    grupo_btc = df_simulado[df_simulado["Criptomoeda"] == "BTC"]
    esperado_btc = grupo_btc[["Valor Real", "Valor Previsto"]].corr().iloc[0, 1]

    correl_btc = df_correlacao[
        (df_correlacao["Criptomoeda"] == "BTC") & (df_correlacao["Modelo"] == "MLP")
    ]["Correlacao"].values[0]

    assert np.isclose(correl_btc, esperado_btc, atol=1e-6), "Correlação BTC/MLP incorreta"
