import pandas as pd
import tempfile
import os
from src.hypothesis import executar_teste_hipotese_retorno_diario_real

def test_executar_teste_hipotese_retorno_diario_real() -> None:
    """
    Testa a função `executar_teste_hipotese_retorno_diario_real` com um DataFrame simulado,
    garantindo que ela retorne um DataFrame com colunas esperadas e dados consistentes.
    """
    # Simula dados de capital diário
    dados = {
        "Data": pd.date_range(start="2023-01-01", periods=5, freq="D"),
        "Criptomoeda": ["TESTCOIN"] * 5,
        "Modelo": ["TEST_MODEL"] * 5,
        "CapitalFinal": [10000, 10100, 10200, 10150, 10300]
    }
    df_simulado = pd.DataFrame(dados)

    # Cria arquivo temporário CSV
    with tempfile.TemporaryDirectory() as tempdir:
        caminho_csv = os.path.join(tempdir, "evolucao_lucro_diario.csv")
        df_simulado.to_csv(caminho_csv, index=False)

        # Executa o teste de hipótese
        df_resultado = executar_teste_hipotese_retorno_diario_real(
            caminho_csv=caminho_csv,
            retorno_esperado=0.1,
            nivel_significancia=0.05,
            salvar_csv=False
        )

    # Validações
    assert isinstance(df_resultado, pd.DataFrame)
    assert not df_resultado.empty
    colunas_esperadas = [
        "Criptomoeda", "Modelo", "Média Retorno (%)", "Retorno Esperado (%)",
        "N dias", "Estatística t", "p-valor", "Rejeita H₀ (médio ≥ x%)"
    ]
    for coluna in colunas_esperadas:
        assert coluna in df_resultado.columns


def test_amostra_menor_que_dois_ignorada(tmp_path) -> None:
    """
    Testa se o grupo com menos de 2 valores de capital é ignorado corretamente.
    """
    df = pd.DataFrame({
        "Data": ["2023-01-01"],
        "Criptomoeda": ["FAKECOIN"],
        "Modelo": ["SIMPLES"],
        "CapitalFinal": [10000]
    })

    caminho = tmp_path / "dados.csv"
    df.to_csv(caminho, index=False)

    df_resultado = executar_teste_hipotese_retorno_diario_real(
        caminho_csv=str(caminho),
        salvar_csv=False
    )

    assert df_resultado.empty, "O resultado deve ser vazio para amostras com menos de 2 valores."
    

def test_salvar_csv_resultado(tmp_path) -> None:
    """
    Testa se o arquivo CSV de resultado é salvo corretamente.
    """
    df = pd.DataFrame({
        "Data": pd.date_range("2023-01-01", periods=3),
        "Criptomoeda": ["XTEST"] * 3,
        "Modelo": ["MODEL_X"] * 3,
        "CapitalFinal": [10000, 10200, 10100]
    })

    caminho_dados = tmp_path / "dados.csv"
    caminho_saida = tmp_path / "resultado.csv"
    df.to_csv(caminho_dados, index=False)

    df_resultado = executar_teste_hipotese_retorno_diario_real(
        caminho_csv=str(caminho_dados),
        salvar_csv=True,
        caminho_saida=str(caminho_saida)
    )

    assert caminho_saida.exists(), "O arquivo CSV de saída não foi criado."
    assert not df_resultado.empty, "O DataFrame de resultado não deveria estar vazio."

