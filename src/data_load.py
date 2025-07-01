import pandas as pd
import logging
from pathlib import Path
from typing import Optional

# Configuração do logger
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def carregar_criptomoeda(caminho_arquivo: str, nome_cripto: str) -> Optional[pd.DataFrame]:
    """
    Carrega e processa um arquivo CSV com os dados de uma criptomoeda.

    Args:
        caminho_arquivo (str): Caminho absoluto ou relativo do arquivo CSV.
        nome_cripto (str): Nome da criptomoeda (usado para log).

    Returns:
        Optional[pd.DataFrame]: DataFrame com os dados limpos, ou None se ocorrer erro.
    """
    try:
        df = pd.read_csv(caminho_arquivo, skiprows=1)

        # Renomeia colunas padrão
        df = df.rename(columns={
            'date': 'Data',
            'open': 'Abertura',
            'high': 'Alta',
            'low': 'Baixa',
            'close': 'Fechamento',
            'volume': 'Volume',
            'volume.1': 'VolumeMoeda'
        })

        # Converte e ordena as datas
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        df.dropna(subset=['Data'], inplace=True)
        df = df.sort_values(by='Data').reset_index(drop=True)

        # Seleciona colunas relevantes
        colunas_base = ['Data', 'Abertura', 'Alta', 'Baixa', 'Fechamento']
        colunas_opcionais = ['Volume', 'VolumeMoeda']
        colunas_presentes = [col for col in colunas_base + colunas_opcionais if col in df.columns]

        df = df[colunas_presentes]

        logging.info(f"[{nome_cripto}] {df.shape[0]} registros carregados com sucesso.")
        return df

    except Exception as e:
        logging.error(f"[{nome_cripto}] Erro ao carregar dados: {e}")
        return None


def carregar_multiplas_criptomoedas(pasta: str) -> dict[str, pd.DataFrame]:
    """
    Carrega todos os arquivos .csv de uma pasta contendo dados de criptomoedas.

    Args:
        pasta (str): Caminho da pasta com arquivos .csv.

    Returns:
        dict[str, pd.DataFrame]: Dicionário com o nome da criptomoeda como chave
                                 e o DataFrame correspondente como valor.
    """
    resultados = {}
    pasta_path = Path(pasta)

    for arquivo in pasta_path.glob("*.csv"):
        nome = arquivo.stem.replace("Poloniex_", "").replace("_d", "")
        df = carregar_criptomoeda(str(arquivo), nome)
        if df is not None:
            resultados[nome] = df

    logging.info(f"{len(resultados)} criptomoedas carregadas com sucesso.")
    return resultados
