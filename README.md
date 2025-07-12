# Projeto Final - Previsão de Criptomoedas com Regressão

Este projeto tem como objetivo aplicar diferentes modelos de regressão para prever retornos de criptomoedas, avaliando o desempenho de cada modelo com base em métricas de erro e retorno simulado. O sistema também realiza testes de hipótese estatística, geração de gráficos, e validação cruzada.

## Estrutura do Projeto

```
PROJETO_FINAL/
├── data/                     # Arquivos de dados de entrada (ex: CSV de preços de criptomoedas)
├── figures/                  # Figuras e gráficos gerados
├── htmlcov/                  # Relatórios de cobertura de testes (pytest-cov)
├── logs/                     # Logs de execução
├── modelos/                  # Modelos treinados salvos (joblib)
├── results/                  # Resultados das simulações e comparações de modelos
├── scripts/                  # Scripts auxiliares para visualizações e testes estatísticos
├── src/                      # Código-fonte principal
│   ├── data_load.py
│   ├── evaluation.py
│   ├── features.py
│   ├── hypothesis.py
│   ├── logging_config.py
│   ├── main.py
│   ├── models.py
│   ├── utils.py
│   └── visualization.py
├── tests/                    # Testes automatizados (pytest)
├── tmp/                      # Arquivos temporários e de controle
├── .coverage                 # Arquivo de cobertura
├── .coveragerc               # Configuração da cobertura
├── .gitignore
├── README.md
├── requirements.txt
└── limpar_modelos.py         # Script para limpar modelos treinados
```

## Requisitos

Os principais pacotes utilizados são:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`
- `scipy`
- `joblib`
- `pytest`
- `pytest-cov`
- `hydra-core`

Instale com:

   ```bash
   pip install -r requirements.txt
   ```

## Execução

Para executar o pipeline completo:

```bash
python main.py --todas --forcar-treinamento
```

Outras opções estão disponíveis via argparse (ex: `--model`, `--threshold`, `--criptomoeda`).

## Testes

Execute os testes com:

```bash
pytest --cov=src tests/
```

Gere e visualize a cobertura com:

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # no Mac
```

## Autor

Camilla Rodrigues – Pós-graduação em Inteligência Artificial Aplicada – IFG