# 📈 Previsão de Preços de Criptomoedas com Redes Neurais e Regressão

Trabalho Final — Módulo I da Pós-Graduação em Inteligência Artificial Aplicada (IFG)

## 🔍 Objetivo

Desenvolver modelos de previsão do preço de fechamento de criptomoedas utilizando:
- Redes neurais (MLP)
- Modelos de regressão linear e polinomial

Além da previsão, são realizadas análises estatísticas, testes de hipótese, avaliação de desempenho por simulação de investimento e testes automatizados.

---

## 📁 Estrutura do Projeto

```
.
├── data/                  # Base de dados de criptomoedas
├── figures/               # Gráficos gerados automaticamente
├── modelos/               # Modelos treinados salvos (joblib)
├── results/               # Resultados das simulações e comparações de modelos
├── scripts/               # Scripts auxiliares (anova, correlação, erro padrão, etc.)
├── src/
│   ├── data_load.py       # Carregamento e limpeza dos dados
│   ├── evaluation.py      # Simulação de investimentos
│   ├── features.py        # Engenharia de features
│   ├── hypothesis.py      # Teste de hipótese
│   ├── main.py            # Script principal com CLI (argparse)
│   ├── models.py          # Modelos de regressão e rede neural (MLP)
│   ├── utils.py           # Salvar modelos, carregar modelos, pre-processamento de dados, etc.
│   └── visualization.py   # Geração de gráficos
├── tests/                 # Testes automatizados (pytest)
├── requirements.txt       # Dependências do projeto
└── README.md              # Instruções e documentação
```

---

## ⚙️ Instalação

```bash
# Clonar o repositório
git clone git@github.com:camillara/Projeto_Final.git

# Criar ambiente virtual e instalar dependências
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
pip install -r requirements.txt
```

---

## 🚀 Como Executar

### ✅ 1. Treinar modelos e simular investimentos
```bash
python3 -m src.main --todas --treinar_modelos --simular --threshold 0.015 --forcar_treinamento
```

### 📊 2. Estatísticas descritivas (Exercício 3: A–D)
```bash
python3 -m src.main --analise_completa --crypto BCHUSDT
```

### 📌 3. Diagramas de dispersão com todos os modelos (Exercício 9A)
```bash
python3 -m src.main --comparar_modelos
```

### 🔁 4. Correlação entre criptos (Exercício 9B)
```bash
PYTHONPATH=. python3 scripts/gerar_correlacoes.py
```

### 🧮 5. Equação dos regressores (Exercício 9C)
```bash
python3 -m src.main --gerar_equacoes --grau-min 2 --grau-max 9
```

### 📏 6. Cálculo do erro padrão (Exercício 9D)
```bash
python3 -m src.main --gerar-erro-padrao
```

### 🤖 7. Comparar erro MLP vs melhor regressor (Exercício 9E)
```bash
PYTHONPATH=. python3 scripts/calcular_erro_padrao_mlp.py
```

### 💰 8. Gráfico de lucro por modelo (Exercício 9F)
```bash
PYTHONPATH=. python3 scripts/gerar_graficos_lucro_obtido_por_modelo.py
```

### 📈 9. Teste de hipótese com retorno esperado (Exercício 10)
```bash
python3 -m src.main --testar-hipotese-retorno --retorno-esperado 0.3 --gerar-graficos
```

### 📊 10. ANOVA — Retorno médio por cripto (Exercício 11A)
```bash
python3 -m src.main --anova-retorno
```

### 🧪 11. ANOVA — Agrupamento por características (Exercício 11B)
```bash
python3 -m src.main --anova-grupo-retorno
```

---

## 🧪 Testes Automatizados

Execute os testes com relatório de cobertura:

```bash
pytest --cov=src --cov-report=term --cov-report=html tests/
open htmlcov/index.html  # ou explorer htmlcov/index.html no Windows
```

---

## 📌 Boas Práticas Adotadas

- ✅ Modularização do código em `src/`
- ✅ Docstrings e `type hints` em todas as funções
- ✅ Logging para tratamento de erros
- ✅ Testes automatizados com `pytest`
- ✅ Validação com `K-Fold Cross-Validation`
- ✅ Código formatado com `black` e validado com `ruff`
- ✅ Gráficos salvos em `figures/` com resolução de 150 dpi

---

## 🧠 Tecnologias Utilizadas

- Python 3.10+
- Scikit-learn
- Pandas / NumPy
- Seaborn / Matplotlib
- Pytest + pytest-cov
- argparse

---

## 👨‍🏫 Professores

- Dr. Eduardo Noronha  
- Me. Otávio Calaça  
- Dr. Eder Brito

## Autora

Camilla Rodrigues – Pós-graduação em Inteligência Artificial Aplicada – IFG