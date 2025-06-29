import argparse
from src.data_load import carregar_multiplas_criptomoedas
from src.features import adicionar_features_basicas
from src.visualization import (
    plot_boxplot,
    plot_histograma,
    plot_linha_media_mediana_moda,
)
from src.models import treinar_modelos


def main():
    parser = argparse.ArgumentParser(
        description="Análise e processamento de dados de criptomoedas"
    )

    parser.add_argument(
        "--crypto",
        type=str,
        help="Nome da criptomoeda (ex: BTCUSDT)",
        required=True
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Exibir os primeiros registros"
    )

    parser.add_argument(
        "--com_features",
        action="store_true",
        help="Aplicar features como retorno diário e média móvel"
    )

    parser.add_argument(
        "--graficos",
        action="store_true",
        help="Gerar gráficos estatísticos (boxplot, histograma, linha do tempo)"
    )

    parser.add_argument(
        "--treinar_modelos",
        action="store_true",
        help="Treinar modelos (MLP, Linear, Polinomial) com validação cruzada"
    )

    args = parser.parse_args()

    # Carrega todos os arquivos da pasta data/
    dados = carregar_multiplas_criptomoedas("data")

    # Busca o DataFrame da cripto selecionada
    df = dados.get(args.crypto.upper())

    if df is None:
        print(f"[ERRO] Criptomoeda '{args.crypto}' não encontrada.")
        return

    # Aplica features, se solicitado
    if args.com_features:
        df = adicionar_features_basicas(df)

    # Exibe os registros, se solicitado
    if args.show:
        print(df.head())

    # Gera gráficos, se solicitado
    if args.graficos:
        print(f"[INFO] Gerando gráficos para {args.crypto.upper()}...")
        plot_boxplot(df, args.crypto.upper())
        plot_histograma(df, args.crypto.upper())
        plot_linha_media_mediana_moda(df, args.crypto.upper())
        print("[OK] Gráficos salvos em figures/")

    # Treina modelos, se solicitado
    if args.treinar_modelos:
        print(f"\n[INFO] Treinando modelos para {args.crypto.upper()}...\n")
        resultados = treinar_modelos(df)

        print("[RESULTADOS - MÉDIA DE ERRO QUADRÁTICO (MSE) - KFold]")
        for nome, info in resultados.items():
            print(f"{nome}: MSE médio = {info['mse']:.4f}")


if __name__ == "__main__":
    main()
