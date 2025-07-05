from src.data_load import carregar_multiplas_criptomoedas

def test_carregar_multiplas_criptomoedas():
    dados = carregar_multiplas_criptomoedas("data")
    assert isinstance(dados, dict)
    assert len(dados) > 0
    for nome, df in dados.items():
        assert not df.empty
        assert "Fechamento" in df.columns
