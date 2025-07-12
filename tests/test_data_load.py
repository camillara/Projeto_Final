import pandas as pd
from src.data_load import carregar_multiplas_criptomoedas

def test_carregar_multiplas_criptomoedas() -> None:
    """
    Testa a função `carregar_multiplas_criptomoedas` verificando se:
    - Retorna um dicionário.
    - Contém pelo menos uma criptomoeda carregada.
    - Cada DataFrame não está vazio e contém a coluna 'Fechamento'.
    """
    dados: dict[str, pd.DataFrame] = carregar_multiplas_criptomoedas("data")

    assert isinstance(dados, dict), "A função deve retornar um dicionário"
    assert len(dados) > 0, "Nenhuma criptomoeda foi carregada"

    for nome, df in dados.items():
        assert isinstance(df, pd.DataFrame), f"O valor para {nome} não é um DataFrame"
        assert not df.empty, f"O DataFrame para {nome} está vazio"
        assert "Fechamento" in df.columns, f"'Fechamento' não encontrada em {nome}"
