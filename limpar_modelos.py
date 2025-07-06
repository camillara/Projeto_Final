import os

# Caminho para a pasta onde os modelos estão salvos
pasta_modelos = "modelos"

# Verifica se a pasta existe
if os.path.exists(pasta_modelos):
    arquivos_removidos = 0
    for nome_arquivo in os.listdir(pasta_modelos):
        if nome_arquivo.endswith(".joblib"):
            caminho_completo = os.path.join(pasta_modelos, nome_arquivo)
            os.remove(caminho_completo)
            print(f"Removido: {nome_arquivo}")
            arquivos_removidos += 1
    if arquivos_removidos == 0:
        print("Nenhum arquivo .joblib encontrado para remover.")
    else:
        print(f"Total de arquivos .joblib removidos: {arquivos_removidos}")
else:
    print(f"A pasta '{pasta_modelos}' não foi encontrada.")
