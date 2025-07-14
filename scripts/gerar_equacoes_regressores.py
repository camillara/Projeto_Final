import os
import gc
import joblib
import logging
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from src.logging_config import configurar_logging
from src.utils import preprocessar_dados


def gerar_equacoes(grau_min: int = 1, grau_max: int = 10) -> None:
    """
    Gera equações matemáticas dos modelos de regressão treinados para diferentes graus polinomiais.

    A função percorre todas as criptomoedas da lista pré-definida, carrega os modelos já treinados
    (lineares e polinomiais), extrai os coeficientes de regressão e monta a equação preditiva no formato:

        y = b0 + b1*x1 + b2*x2 + ... + bn*xn

    Os modelos polinomiais utilizam transformação com PolynomialFeatures para representar os termos
    de maior ordem.

    Parameters:
    -----------
    grau_min : int, optional
        Grau mínimo do polinômio a ser processado (default é 1, que representa o modelo Linear).
    grau_max : int, optional
        Grau máximo do polinômio a ser processado (default é 5).

    Outputs:
    --------
    - Um arquivo CSV será salvo em `results/equacoes_regressores_grau_<grau_min>_a_<grau_max>.csv`,
      contendo as colunas:
        - Criptomoeda
        - Modelo (ex: Linear, Polinomial_2, etc.)
        - Equacao (string formatada com coeficientes e variáveis)

    Observações:
    ------------
    - Modelos devem estar previamente salvos na pasta `modelos/`.
    - Os dados originais devem estar disponíveis na pasta `data/`, com prefixo `Poloniex_`.
    - O log de execução será salvo em `logs/gerar_equacoes_regressores.log`.
    - A função utiliza garbage collection (`gc.collect()`) ao final de cada modelo para reduzir uso de memória.

    Exemplo de uso:
    ---------------
    >>> gerar_equacoes(grau_min=1, grau_max=5)
    """

    os.makedirs("logs", exist_ok=True)
    configurar_logging("logs/gerar_equacoes_regressores.log")
    logging.info(
        f"Iniciando geração das equações dos regressores (grau {grau_min} a {grau_max})..."
    )

    DIRETORIO_MODELOS = "modelos"
    DIRETORIO_DADOS = "data"
    DIRETORIO_SAIDA = "results"
    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

    # Lista padrão de criptomoedas
    criptos_padrao = [
        "DASHUSDT",
        "XRPUSDT",
        "XMRUSDT",
        "ETHUSDT",
        "LTCUSDT",
        "ZRXUSDT",
        "BTCUSDT",
        "BATUSDT",
        "BCHUSDT",
        "ETCUSDT",
    ]

    criptos = criptos_padrao

    equacoes = []

    for cripto in criptos:
        logging.info(f"\n[CRYPTO] Processando {cripto}...")

        caminho_csv = os.path.join(DIRETORIO_DADOS, f"Poloniex_{cripto}_d.csv")
        if not os.path.exists(caminho_csv):
            logging.warning(f"Arquivo CSV não encontrado: {caminho_csv}")
            continue

        try:
            df = pd.read_csv(caminho_csv, skiprows=1)
            df = preprocessar_dados(df)
            X = df.drop(columns=["Fechamento", "Data"], errors="ignore")
            feature_names = X.columns.tolist()
        except Exception as e:
            logging.error(f"Erro ao processar dados de {cripto}: {e}")
            continue

        for grau in range(grau_min, grau_max + 1):
            nome_modelo = (
                f"{cripto}_linear.joblib"
                if grau == 1
                else f"{cripto}_polinomial_grau{grau}.joblib"
            )
            modelo_path = os.path.join(DIRETORIO_MODELOS, nome_modelo)

            if not os.path.exists(modelo_path):
                logging.warning(f"Modelo não encontrado: {modelo_path}")
                continue

            try:
                modelo = joblib.load(modelo_path)

                if grau == 1:
                    coef = modelo.coef_
                    intercept = modelo.intercept_
                    nomes_features = feature_names
                else:
                    poly = PolynomialFeatures(degree=grau)
                    poly.fit(X)
                    nomes_features = poly.get_feature_names_out(feature_names)
                    coef = modelo.named_steps["linearregression"].coef_
                    intercept = modelo.named_steps["linearregression"].intercept_

                termos = [
                    f"{coef[i]:.4f}*{nomes_features[i]}" for i in range(len(coef))
                ]
                equacao = " + ".join(termos)
                equacao = f"y = {intercept:.4f} + " + equacao

                equacoes.append(
                    {
                        "Criptomoeda": cripto,
                        "Modelo": "Linear" if grau == 1 else f"Polinomial_{grau}",
                        "Equacao": equacao,
                    }
                )

                logging.info(f"✔️ Equação gerada: {cripto} - Grau {grau}")
            except Exception as e:
                logging.error(f"Erro ao gerar equação para {cripto} - Grau {grau}: {e}")
            finally:
                del modelo, coef, intercept, nomes_features
                if grau != 1:
                    del poly
                gc.collect()

    df_equacoes = pd.DataFrame(equacoes)
    saida = os.path.join(DIRETORIO_SAIDA, "equacoes_regressores.csv")

    # Se o arquivo já existir, carregar conteúdo anterior
    if os.path.exists(saida):
        try:
            df_existente = pd.read_csv(saida)
            df_equacoes = pd.DataFrame(equacoes)
            df_total = pd.concat([df_existente, df_equacoes], ignore_index=True)
            df_total.drop_duplicates(subset=["Criptomoeda", "Modelo"], inplace=True)
        except Exception as e:
            logging.warning(
                f"Não foi possível ler o CSV existente. Criando novo arquivo. Erro: {e}"
            )
            df_total = pd.DataFrame(equacoes)
    else:
        df_total = pd.DataFrame(equacoes)

    df_total.to_csv(saida, index=False)
    logging.info(f"[OK] Equações salvas/atualizadas em {saida}")
