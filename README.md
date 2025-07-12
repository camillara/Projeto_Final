# Projeto Final - Inteligência Artificial Aplicada

Este projeto tem como objetivo analisar o desempenho de diferentes modelos de regressão para prever os retornos de criptomoedas, utilizando métricas de erro e estratégias de simulação de lucro. O projeto foi desenvolvido como parte do curso de pós-graduação em Inteligência Artificial Aplicada do IFG.

## Estrutura do Projeto

- `src/`: Contém os módulos principais do projeto.
  - `data_load.py`: Funções para carregamento de dados de criptomoedas.
  - `features.py`: Adição de features como média móvel e tendência.
  - `models.py`: Treinamento dos modelos de regressão.
  - `evaluation.py`: Avaliação dos modelos com simulação de investimento.
  - `visualization.py`: Geração de gráficos e análises estatísticas.
  - `utils.py`: Funções auxiliares.
  - `logging_config.py`: Configuração de logs.
  - `hypothesis.py`: Testes de hipótese.

- `scripts/`: Scripts complementares para análise estatística e geração de gráficos.
- `tests/`: Testes automatizados com `pytest`.
- `data/`: Pasta esperada para conter os dados CSV de criptomoedas.
- `figures/`: Saída dos gráficos gerados.
- `modelos/`: Armazenamento dos modelos treinados.
- `htmlcov/`: Relatório de cobertura de testes.

## Instalação

1. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # ou venv\Scripts\activate no Windows
   ```

2. Instale as dependências:
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
