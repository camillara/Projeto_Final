import argparse
import pandas as pd
from src.data_load import carregar_multiplas_criptomoedas
from src.features import adicionar_features_basicas
from src.visualization import (
    plot_boxplot,
    plot_histograma,
    plot_linha_media_mediana_moda,
    calcular_dispersao
)
from src.models import treinar_modelos
from src.evaluation import simular_estrategia_investimento
from src.utils import plot_grafico_retorno


def main():
    parser = argparse.ArgumentParser(
        description="Análise e processamento de dados de criptomoedas"
    )

    parser.add_argument("--crypto", type=str, help="Nome da criptomoeda (ex: BTCUSDT)")
    parser.add_argument("--show", action="store_true", help="Exibir os primeiros registros")
    parser.add_argument("--com_features", action="store_true", help="Aplicar features como retorno diário e média móvel")
    parser.add_argument("--graficos", action="store_true", help="Gerar gráficos estatísticos")
    parser.add_argument("--treinar_modelos", action="store_true", help="Treinar modelos (MLP, Linear, Polinomial)")
    parser.add_argument("--simular", action="store_true", help="Simular estratégia de investimento com MLP")
    parser.add_argument("--threshold", type=float, help="Valor mínimo de previsão para decidir compra (ex: 0.01)")
    parser.add_argument("--todas", action="store_true", help="Executar simulações para todas as criptomoedas")
    parser.add_argument("--forcar_treinamento", action="store_true", help="Forçar re-treinamento mesmo que modelos já existam")
    parser.add_argument("--analise_completa", action="store_true", help="Executar análise gráfica e estatística completa para todas as criptomoedas")
    parser.add_argument("--model", type=str, help="Nome do modelo a ser usado (MLP, Linear, Polinomial)")
    parser.add_argument("--kfolds", type=int, default=5, help="Número de folds para validação cruzada KFold (padrão = 5)")

    args = parser.parse_args()

    dados = carregar_multiplas_criptomoedas("data")

    if args.todas:
        print("[INFO] Executando para todas as criptomoedas da pasta /data...\n")
        resultados_simulacoes = []

        for nome, df in dados.items():
            print(f"\nProcessando {nome}...")

            if args.com_features:
                df = adicionar_features_basicas(df)

            resultados = treinar_modelos(
                df,
                nome_cripto=args.crypto.upper(),
                reutilizar=not args.forcar_treinamento,
                modelo_especifico=args.model,
                num_folds=args.kfolds
            )
            mlp_model = resultados.get("MLP", {}).get("modelo")
            mse_mlp = resultados.get("MLP", {}).get("mse")

            if mlp_model:
                colunas_remover = ["Fechamento", "Data"]
                X_final = df.drop(columns=[col for col in colunas_remover if col in df.columns])
                y_final = df["Fechamento"]

                mlp_model.fit(X_final, y_final)
                y_pred = mlp_model.predict(X_final)

                _, df_simulacao = simular_estrategia_investimento(df, y_pred, threshold=args.threshold or 0.01)

                lucro_final = df_simulacao["CapitalFinal"].iloc[-1]
                retorno_percentual = df_simulacao["RetornoPercentual"].iloc[-1]
            else:
                lucro_final = None
                retorno_percentual = None

            # Cálculo das medidas estatísticas
            media = df["Fechamento"].mean()
            mediana = df["Fechamento"].median()
            moda = df["Fechamento"].mode().iloc[0] if not df["Fechamento"].mode().empty else None
            desvio_padrao = df["Fechamento"].std()
            variancia = df["Fechamento"].var()
            coef_var = (desvio_padrao / media) * 100 if media else None

            resultados_simulacoes.append({
                "Criptomoeda": nome,
                "MSE_MLP": mse_mlp,
                "LucroFinal": lucro_final,
                "RetornoPercentual": retorno_percentual,
                "Média": media,
                "Mediana": mediana,
                "Moda": moda,
                "Desvio Padrão": desvio_padrao,
                "Variância": variancia,
                "Coef. Variação (%)": coef_var
            })

        df_resultados = pd.DataFrame(resultados_simulacoes)
        df_resultados.sort_values(by="RetornoPercentual", ascending=False, inplace=True)
        df_resultados.to_csv("resultados_simulacoes.csv", index=False)
        print("\n[OK] Resultados salvos em resultados_simulacoes.csv")

        plot_grafico_retorno(df_resultados)
        print("[OK] Gráfico salvo em figures/retornos_criptos.png")
        return

    if not args.crypto:
        print("[ERRO] Informe --crypto ou use --todas.")
        return

    df = dados.get(args.crypto.upper())
    if df is None:
        print(f"[ERRO] Criptomoeda '{args.crypto}' não encontrada.")
        return

    if args.com_features:
        df = adicionar_features_basicas(df)

    if args.show:
        print(df.head())

    if args.graficos:
        print(f"[INFO] Gerando gráficos para {args.crypto.upper()}...")
        plot_boxplot(df, args.crypto.upper())
        plot_histograma(df, args.crypto.upper())
        plot_linha_media_mediana_moda(df, args.crypto.upper())
        calcular_dispersao(df, args.crypto.upper())
        print("[OK] Gráficos salvos em figures/")

    if args.treinar_modelos:
        print(f"\n[INFO] Treinando modelos para {args.crypto.upper()}...\n")
        resultados = treinar_modelos(
            df,
            nome_cripto=args.crypto.upper(),
            reutilizar=not args.forcar_treinamento,
            modelo_especifico=args.model,
            num_folds=args.kfolds
        )

        print("[RESULTADOS - MÉDIA DE ERRO QUADRÁTICO (MSE) - KFold]")
        for nome, info in resultados.items():
            mse = info['mse']
            print(f"{nome}: MSE médio = {mse:.4f}" if mse is not None and not pd.isna(mse) else f"{nome}: modelo carregado (MSE não reavaliado)")

        if args.simular:
            print(f"\n[INFO] Simulando estratégia com MLP para {args.crypto.upper()}...\n")
            mlp_model = resultados.get("MLP", {}).get("modelo")

            if mlp_model:
                colunas_remover = ["Fechamento", "Data"]
                X_final = df.drop(columns=[col for col in colunas_remover if col in df.columns])
                y_final = df["Fechamento"]

                mlp_model.fit(X_final, y_final)
                y_pred = mlp_model.predict(X_final)

                _, df_simulacao = simular_estrategia_investimento(df, y_pred, threshold=args.threshold or 0.01)
            else:
                print("[ERRO] Modelo MLP não encontrado nos resultados.")

    if args.analise_completa:
        print("[INFO] Executando análise gráfica e estatística para todas as criptomoedas...\n")
        for nome, df in dados.items():
            print(f"\nGerando gráficos e estatísticas para {nome}...")

            plot_boxplot(df, nome)
            plot_histograma(df, nome)
            plot_linha_media_mediana_moda(df, nome)
            calcular_dispersao(df, nome)

        print("\n[OK] Gráficos salvos em figures/, medidas de dispersão no log.")
        return


if __name__ == "__main__":
    main()
