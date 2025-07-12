import argparse
import os
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
from src.hypothesis import testar_hipotese_retorno_diario_real
from scripts.gerar_equacoes_regressores import gerar_equacoes
from scripts.erro_padrao import calcular_erro_padrao
from scripts.gerar_graficos_erro_padrao import gerar_graficos_erro_padrao
from scripts.plot_testes_hipotese import plotar_graficos_teste_hipotese
from scripts.anova_retorno_criptos import executar_anova
from scripts.anova_por_grupo_caracteristica import executar_anova_por_grupo

def main():
    parser = argparse.ArgumentParser(
        description="Análise e processamento de dados de criptomoedas"
    )

    parser.add_argument("--crypto", type=str, help="Nome da criptomoeda (ex: BTCUSDT)")
    parser.add_argument("--show", action="store_true", help="Exibir os primeiros registros")
    parser.add_argument("--graficos", action="store_true", help="Gerar gráficos estatísticos")
    parser.add_argument("--treinar_modelos", action="store_true", help="Treinar modelos (MLP, Linear, Polinomial)")
    parser.add_argument("--simular", action="store_true", help="Simular estratégia de investimento com todos os modelos")
    parser.add_argument("--threshold", type=float, help="Valor mínimo de previsão para decidir compra (ex: 0.01)")
    parser.add_argument("--todas", action="store_true", help="Executar simulações para todas as criptomoedas")
    parser.add_argument("--forcar_treinamento", action="store_true", help="Forçar re-treinamento mesmo que modelos já existam")
    parser.add_argument("--analise_completa", action="store_true", help="Executar análise gráfica e estatística completa para todas as criptomoedas")
    parser.add_argument("--model", type=str, help="Nome do modelo a ser usado (MLP, Linear, Polinomial)")
    parser.add_argument("--kfolds", type=int, default=5, help="Número de folds para validação cruzada KFold (padrão = 5)")
    parser.add_argument("--comparar_modelos", action="store_true", help="Executar comparação de desempenho entre os modelos")
    parser.add_argument("--gerar_equacoes", action="store_true", help="Gerar equações dos modelos regressivos")
    parser.add_argument("--grau-min", type=int, default=1, help="Grau mínimo")
    parser.add_argument("--grau-max", type=int, default=5, help="Grau máximo")
    parser.add_argument("--gerar-erro-padrao", action="store_true", help="Gera o erro padrão (RMSE) para cada cripto e modelo.")
    parser.add_argument("--graficos-erro-padrao", "--graficos_erro_padrao", dest="graficos_erro_padrao", action="store_true", help="Gera os gráficos de erro padrão.")
    parser.add_argument("--testar-hipotese-retorno", action="store_true", help="Executa o teste t de hipótese sobre o retorno médio diário.")
    parser.add_argument("--retorno-esperado", type=float, default=0.1, help="Valor de retorno médio diário esperado (em %), ex: 0.1 para 0.1%.")
    parser.add_argument("--gerar-graficos", action="store_true", help="Gera gráficos com os resultados do teste de hipótese.")
    parser.add_argument("--anova-retorno", action="store_true", help="Executa ANOVA para comparar retornos médios entre criptomoedas.")
    parser.add_argument("--anova-grupo-retorno", action="store_true", help="Executa ANOVA entre grupos por retorno médio diário.")
    
    args = parser.parse_args()

    dados = carregar_multiplas_criptomoedas("data")
    
    if args.gerar_equacoes:
        gerar_equacoes(grau_min=args.grau_min, grau_max=args.grau_max)
        return 
    
    
    if args.gerar_erro_padrao:
        calcular_erro_padrao()
        return 
    
    if args.graficos_erro_padrao:
        gerar_graficos_erro_padrao()
        return
    

    if args.testar_hipotese_retorno:
        print("\n[INFO] Executando teste de hipótese sobre o retorno médio diário...\n")
        resultado_hipotese = testar_hipotese_retorno_diario_real(
            retorno_esperado=args.retorno_esperado,
            salvar_csv=True
        )
        print(resultado_hipotese.to_string(index=False))

        if args.gerar_graficos:
            print("\n[INFO] Gerando gráficos do teste de hipótese...\n")
            plotar_graficos_teste_hipotese()
        
        return
    
    
    if args.anova_retorno:
        caminho_csv = "results/teste_hipotese_retorno_diario.csv"
        pasta_saida = "results"
        os.makedirs(pasta_saida, exist_ok=True)
        executar_anova(caminho_csv, pasta_saida)
        return


    if args.anova_grupo_retorno:
        caminho_csv = os.path.join("results", "teste_hipotese_retorno_diario.csv")
        pasta_saida = "results/anova_grupo_retorno"
        executar_anova_por_grupo(caminho_csv, pasta_saida, coluna_agrupadora="Média Retorno (%)")
        return


    if args.todas:
        print("[INFO] Executando para todas as criptomoedas da pasta /data...\n")
        resultados_simulacoes = []
        previsto_real_geral = []
        evolucao_diaria_lucro = []

        for nome, df in dados.items():
            print(f"\nProcessando {nome}...")

            resultados = treinar_modelos(
                df,
                nome_cripto=nome,
                reutilizar=not args.forcar_treinamento,
                modelo_especifico=args.model,
                num_folds=args.kfolds
            )

            linha_resultado = {"Criptomoeda": nome}

            for modelo_nome, resultado in resultados.items():
                predicoes = resultado.get("previsoes")
                mse = resultado.get("mse")
                linha_resultado[f"MSE_{modelo_nome}"] = mse

                y_real = resultado.get("y_real")
                y_pred = resultado.get("previsoes")
                if y_real is not None and y_pred is not None:
                    for real, pred in zip(y_real, y_pred):
                        previsto_real_geral.append({
                            "Criptomoeda": nome,
                            "Modelo": modelo_nome,
                            "Valor Real": real,
                            "Valor Previsto": pred
                        })

                if predicoes is not None:
                    lucro_total, df_sim = simular_estrategia_investimento(
                        df, predicoes, threshold=args.threshold or 0.01
                    )
                    linha_resultado[f"RetornoPercentual_{modelo_nome}"] = df_sim["RetornoPercentual"].iloc[-1]
                    linha_resultado[f"Lucro_{modelo_nome}"] = lucro_total

                    # ✅ Adição: salvar evolução diária do lucro
                    if not df_sim.empty and "CapitalFinal" in df_sim.columns:
                        for i, row in df_sim.iterrows():
                            evolucao_diaria_lucro.append({
                                "Data": row.get("Data", f"Dia {i+1}"),
                                "Criptomoeda": nome,
                                "Modelo": modelo_nome,
                                "CapitalFinal": row["CapitalFinal"]
                            })
                else:
                    linha_resultado[f"RetornoPercentual_{modelo_nome}"] = None
                    linha_resultado[f"Lucro_{modelo_nome}"] = None

            # Estatísticas descritivas do preço de fechamento
            linha_resultado.update({
                "Média": df["Fechamento"].mean(),
                "Mediana": df["Fechamento"].median(),
                "Moda": df["Fechamento"].mode().iloc[0] if not df["Fechamento"].mode().empty else None,
                "Desvio Padrão": df["Fechamento"].std(),
                "Variância": df["Fechamento"].var(),
                "Coef. Variação (%)": (df["Fechamento"].std() / df["Fechamento"].mean()) * 100
            })

            resultados_simulacoes.append(linha_resultado)

        df_resultados = pd.DataFrame(resultados_simulacoes)
        col_ordenacao = "RetornoPercentual_MLP" if "RetornoPercentual_MLP" in df_resultados.columns else df_resultados.columns[1]
        df_resultados.sort_values(by=col_ordenacao, ascending=False, inplace=True)
        
        df_evolucao = pd.DataFrame(evolucao_diaria_lucro)
        os.makedirs("results", exist_ok=True)
        
        df_resultados.to_csv("results/resultados_simulacoes.csv", index=False)
        print("\n[OK] Resultados salvos em results/resultados_simulacoes.csv")

        # Salvando previsto vs real
        df_previsto_real = pd.DataFrame(previsto_real_geral)
        df_previsto_real.to_csv("results/previsto_real_por_modelo_por_cripto.csv", index=False)
        print("[OK] Valores reais e previstos salvos em results/previsto_real_por_modelo_por_cripto.csv")

        # Salvando evolução do capital diário
        df_evolucao.to_csv("results/evolucao_lucro_diario.csv", index=False)
        print("[OK] Lucro diário salvo em results/evolucao_lucro_diario.csv")

        if "RetornoPercentual_MLP" in df_resultados.columns:
            plot_grafico_retorno(df_resultados)

        modelos_disponiveis = [col for col in df_resultados.columns if col.startswith("RetornoPercentual_")]
        if len(modelos_disponiveis) > 1:
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
