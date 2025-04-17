# Previsão de Preços de Casas - Explicação Passo a Passo

Este repositório contém um Jupyter Notebook (`HousePricesX.ipynb`) que realiza uma análise de regressão para prever preços de casas usando o conjunto de dados Ames Housing. Abaixo, explico o código passo a passo, incluindo o objetivo de cada seção, os métodos utilizados e os resultados obtidos, como se fosse eu explicando o projeto.

## Visão Geral do Projeto
O objetivo deste projeto é prever preços de casas com base em várias características, como tamanho, qualidade e localização. O notebook utiliza análise exploratória de dados (EDA), pré-processamento de dados e modelos de aprendizado de máquina para alcançar esse objetivo. Três modelos — Regressão Linear, Random Forest e XGBoost — são treinados e comparados para identificar o melhor desempenho.

---

## Explicação Passo a Passo

### 1. Importação de Bibliotecas
Começo importando bibliotecas essenciais do Python para manipulação de dados, visualização e aprendizado de máquina:
- **Pandas** e **NumPy** para manipulação de dados e operações numéricas.
- **Matplotlib** e **Seaborn** para criar visualizações.
- **Scikit-learn** para treinamento de modelos (Regressão Linear, Random Forest) e métricas de avaliação (MAE, RMSE, R²).
- **XGBoost** para o modelo de gradient boosting.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

Também configuro o estilo de visualização usando o tema "whitegrid" do Seaborn e um tamanho padrão para as figuras, garantindo consistência nos gráficos.

### 2. Carregamento do Conjunto de Dados
O conjunto de dados é carregado de um arquivo CSV (`train.csv`) usando o Pandas. Ele contém 1460 linhas e 81 colunas, incluindo características como tamanho do lote, número de quartos e a variável alvo `SalePrice` (preço de venda).

```python
df = pd.read_csv("train.csv")
print("Formato do dataset:", df.shape)
df.head()
```

**Saída**: O conjunto de dados possui 1460 entradas e 81 características, com `SalePrice` como a variável alvo.

### 3. Análise Exploratória de Dados (EDA)
Realizo uma análise exploratória para entender a estrutura do conjunto de dados e identificar padrões ou problemas:
- **Informações do Conjunto de Dados**: Uso `df.info()` para verificar tipos de dados e valores ausentes. O conjunto tem uma mistura de características numéricas (38 colunas) e categóricas (43 colunas), com algumas colunas (como `PoolQC` e `MiscFeature`) apresentando muitos valores ausentes.
- **Valores Ausentes**: Colunas como `PoolQC` (1453 valores ausentes) e `Alley` (1369 valores ausentes) são sinalizadas para pré-processamento.
- **Estatísticas Descritivas**: Uso `df.describe()` para resumir as características numéricas, observando que `SalePrice` varia de $34.900 a $755.000, com média de aproximadamente $180.921.

```python
print("Informações gerais:")
df.info()
print("\nValores ausentes:")
print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False).head(10))
print("\nEstatísticas descritivas:")
df.describe()
```

**Resultados**:
- Muitas características categóricas têm valores ausentes, provavelmente indicando a ausência de uma característica (por exemplo, sem piscina para `PoolQC`).
- Características numéricas como `LotFrontage` e `MasVnrArea` têm valores ausentes moderados que precisam ser tratados.
- `SalePrice` é assimétrico à direita, sugerindo a possibilidade de transformação (embora não aplicada nesta versão).

### 4. Pré-processamento de Dados (Não Totalmente Exibido no Código Fornecido)
Embora o trecho do notebook fornecido não inclua todas as etapas de pré-processamento, normalmente realizo o seguinte (com base em práticas padrão para este conjunto de dados):
- **Valores Ausentes**:
  - Para colunas categóricas como `PoolQC`, imputo valores ausentes com "None" (indicando ausência de piscina).
  - Para colunas numéricas como `LotFrontage`, imputo com a mediana para evitar a influência de outliers.
- **Seleção de Características**: Seleciono características numéricas relevantes (como `GrLivArea`, `OverallQual`, `TotalBsmtSF`) e codifico características categóricas (por exemplo, usando codificação one-hot para `Neighborhood`).
- **Divisão dos Dados**: O conjunto de dados é dividido em conjuntos de treino e teste usando `train_test_split` para avaliar o desempenho do modelo.

**Suposição**: O notebook provavelmente inclui essas etapas antes do treinamento do modelo, já que o código posterior referencia `treino_X`, `teste_X`, `treino_y` e `teste_y`.

### 5. Treinamento e Avaliação dos Modelos
Treino três modelos de regressão para prever `SalePrice`:
- **Regressão Linear**: Um modelo básico que assume relações lineares.
- **Random Forest Regressor**: Um modelo de ensemble que lida com relações não lineares e interações entre características.
- **XGBoost Regressor**: Um modelo de gradient boosting conhecido por alto desempenho em dados estruturados.

Para cada modelo, eu:
- Treino o modelo no conjunto de treino.
- Faço previsões no conjunto de teste.
- Calculo métricas de avaliação:
  - **Erro Absoluto Médio (MAE)**: Média das diferenças absolutas entre preços previstos e reais.
  - **Raiz do Erro Quadrático Médio (RMSE)**: Raiz quadrada da média dos erros quadrados, penalizando erros maiores.
  - **R²**: Proporção da variância em `SalePrice` explicada pelo modelo (quanto mais próximo de 1, melhor).

Também visualizo as previsões usando um gráfico de dispersão dos preços reais versus previstos, com uma linha tracejada vermelha representando previsões perfeitas.

```python
plt.scatter(teste_y, predicoes, alpha=0.6)
plt.xlabel("Preço de Venda Real")
plt.ylabel("Preço de Venda Previsto")
plt.title(f"{nome} - Real x Previsto")
plt.plot([teste_y.min(), teste_y.max()], [teste_y.min(), teste_y.max()], 'r--')
plt.show()
```

As previsões são salvas em um arquivo CSV para análise posterior.

```python
previsoes_df = pd.DataFrame({"Preco_Real": teste_y.values, "Preco_Previsto": predicoes})
previsoes_df.to_csv(f"previsoes_{nome.replace(' ', '_').lower()}.csv", index=False)
```

### 6. Comparação dos Modelos
Compilo os resultados de todos os modelos em um DataFrame para comparar seu desempenho com base em MAE, RMSE e R².

```python
resultado_df = pd.DataFrame(resultados)
print(resultado_df)
```

**Resultados**:
```
             Modelo           MAE          RMSE        R²
0  Regressão Linear  25420.776687  39944.399347  0.791983
1     Random Forest  19993.524931  29605.538515  0.885730
2           XGBoost  20045.656250  29648.797678  0.885396
```

**Análise**:
- **Random Forest** tem o melhor desempenho, com o menor MAE ($19.993) e RMSE ($29.605) e o maior R² (0,886). Isso indica que ele explica 88,6% da variância em `SalePrice` e tem os menores erros de previsão.
- **XGBoost** é muito próximo do Random Forest, com MAE ligeiramente maior ($20.045) e RMSE ($29.648) e um R² um pouco menor (0,885).
- **Regressão Linear** é o mais fraco, com erros maiores (MAE: $25.420, RMSE: $39.944) e um R² menor (0,792), sugerindo dificuldade com relações não lineares nos dados.

### 7. Visualizações
Os gráficos de dispersão para cada modelo mostram o quão bem as previsões se alinham com os preços reais. Pontos mais próximos da linha tracejada vermelha indicam melhores previsões. Random Forest e XGBoost provavelmente mostram uma maior concentração ao redor da linha em comparação com a Regressão Linear, refletindo seu desempenho superior.

---

## Principais Descobertas
- **Melhor Modelo**: Random Forest supera ligeiramente o XGBoost e significativamente a Regressão Linear, sendo a melhor escolha para este conjunto de dados.
- **Importância das Características**: Embora não explicitamente mostrado no código, Random Forest e XGBoost podem fornecer escores de importância das características. Características como `OverallQual` (qualidade geral) e `GrLivArea` (área de estar) geralmente são fortes preditores de preços de casas.
- **Desafios**: Valores ausentes e codificação de variáveis categóricas exigem um pré-processamento cuidadoso. O tamanho do conjunto de dados (1460 linhas) limita a complexidade do modelo, mas Random Forest e XGBoost lidam bem com isso.

## Melhorias Futuras
- **Engenharia de Características**: Criar novas características, como área total ou idade da casa na venda.
- **Ajuste de Hiperparâmetros**: Usar busca em grade ou busca aleatória para otimizar os parâmetros de Random Forest e XGBoost.
- **Validação Cruzada**: Implementar validação cruzada k-fold para estimativas de desempenho mais robustas.
- **Transformação Logarítmica**: Aplicar uma transformação logarítmica em `SalePrice` para lidar com sua assimetria e potencialmente melhorar o desempenho do modelo.

## Como Executar o Notebook
1. Clone este repositório: `git clone <url-do-repositório>`
2. Instale as dependências: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost`
3. Baixe o conjunto de dados (`train.csv`) da competição [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) e coloque-o na pasta do repositório.
4. Execute o notebook usando o Jupyter: `jupyter notebook HousePricesX.ipynb`

## Conclusão
Este projeto demonstra um fluxo completo de aprendizado de máquina, desde a análise exploratória até a comparação de modelos, para prever preços de casas. O Random Forest se destacou como o melhor modelo, oferecendo previsões precisas e robustas. Este trabalho pode ser expandido com técnicas avançadas de engenharia de características e ajustes de hiperparâmetros para melhorar ainda mais os resultados.
