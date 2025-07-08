import argparse
import pandas as pd
from src.data_load import carregar_multiplas_criptomoedas
from src.features import adicionar_features_basicas
from src.visualization import (
    plot_boxplot,
    plot_histograma,
    plot_linha_media_mediana_moda,
    calcular_dispersao,
    plotar_dispersao_e_lucros,
    plot_grafico_comparativo_modelos
)
from src.models import treinar_modelos
from src.evaluation import simular_estrategia_investimento, comparar_modelos_regressao
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
    parser.add_argument("--comparar_modelos", action="store_true", help="Executar comparação de desempenho entre os modelos")


    args = parser.parse_args()

    dados = carregar_multiplas_criptomoedas("data")

    if args.todas:
        print("[INFO] Executando para todas as criptomoedas da pasta /data...\n")
        resultados_simulacoes = []

        for nome, df in dados.items():
            print(f"\nProcessando {nome}...")

            if args.com_features:
                df = adicionar_features_basicas(df)

            # Treina os modelos ou carrega versões anteriores
            resultados = treinar_modelos(
                df,
                nome_cripto=nome,
                reutilizar=not args.forcar_treinamento,
                modelo_especifico=args.model,
                num_folds=args.kfolds
            )

            # Inicializa dicionário para armazenar retornos de cada modelo
            dados_modelos = {}
            for modelo_nome in ["MLP", "Linear", "Polinomial_2"]:
                predicoes = resultados.get(modelo_nome, {}).get("previsoes")
                if predicoes is not None:
                    _, df_sim = simular_estrategia_investimento(df, predicoes, threshold=args.threshold or 0.01)
                    dados_modelos[modelo_nome] = df_sim["RetornoPercentual"].iloc[-1]
                else:
                    dados_modelos[modelo_nome] = None

            # Cálculo das estatísticas descritivas
            media = df["Fechamento"].mean()
            mediana = df["Fechamento"].median()
            moda = df["Fechamento"].mode().iloc[0] if not df["Fechamento"].mode().empty else None
            desvio_padrao = df["Fechamento"].std()
            variancia = df["Fechamento"].var()
            coef_var = (desvio_padrao / media) * 100 if media else None

            # Adiciona ao DataFrame de resultados
            resultados_simulacoes.append({
                "Criptomoeda": nome,
                "MSE_MLP": resultados.get("MLP", {}).get("mse"),
                "RetornoPercentual_MLP": dados_modelos.get("MLP"),
                "RetornoPercentual_Linear": dados_modelos.get("Linear"),
                "MSE_Linear": resultados.get("Linear", {}).get("mse"),
                "RetornoPercentual_Polinomial_2": dados_modelos.get("Polinomial_2"),
                "MSE_Polinomial_2": resultados.get("Polinomial_2", {}).get("mse"),
                "Média": media,
                "Mediana": mediana,
                "Moda": moda,
                "Desvio Padrão": desvio_padrao,
                "Variância": variancia,
                "Coef. Variação (%)": coef_var
            })

        # Gera DataFrame final com os resultados e salva em CSV
        df_resultados = pd.DataFrame(resultados_simulacoes)
        df_resultados.sort_values(by="RetornoPercentual_MLP", ascending=False, inplace=True)
        df_resultados.to_csv("resultados_simulacoes.csv", index=False)
        print("\n[OK] Resultados salvos em resultados_simulacoes.csv")

        # Gera gráfico tradicional da MLP
        plot_grafico_retorno(df_resultados)
        print("[OK] Gráfico salvo em figures/retornos_criptos.png")

        # Gera gráfico comparativo entre modelos, se colunas estiverem disponíveis
        if "RetornoPercentual_Linear" in df_resultados.columns and "RetornoPercentual_Poly_2" in df_resultados.columns:
            plot_grafico_comparativo_modelos(df_resultados)

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
    
    if args.comparar_modelos:
        if args.crypto is None:
            raise ValueError("Para comparar modelos, especifique a criptomoeda com --crypto NOME.")

        print(f"[INFO] Comparando modelos para {args.crypto.upper()}...\n")

        resultados = treinar_modelos(
            df,
            nome_cripto=args.crypto.upper(),
            reutilizar=not args.forcar_treinamento,
            modelo_especifico=None,
            num_folds=args.kfolds
        )

        # Obtém as previsões do MLP
        mlp_modelo = resultados["MLP"]["modelo"]
        X = df.drop(columns=["Fechamento", "Data"], errors="ignore")
        y_real = df["Fechamento"].values
        mlp_preds = mlp_modelo.predict(X)

        # Comparação completa dos modelos
        resultados_completos = comparar_modelos_regressao(df, y_real, mlp_preds)

        plotar_dispersao_e_lucros(resultados_completos)
        print("[OK] Comparação entre modelos finalizada. Gráficos e métricas salvos em /figures.")



if __name__ == "__main__":
    main()
