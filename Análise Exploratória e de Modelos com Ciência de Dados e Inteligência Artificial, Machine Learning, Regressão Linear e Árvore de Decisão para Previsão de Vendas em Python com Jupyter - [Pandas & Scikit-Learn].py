#!/usr/bin/env python
# coding: utf-8

# In[10]:
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install scikit-learn')

# In[11]:
# ---- Importação da Base de Dados
import pandas as pd
tabela = pd.read_csv(r"Endereço da Planilha advertising.csv")
# read_csv: realiza Leitura de Planilha de Base de Dados
display(tabela)
# display: realiza Impressão da Tabela em Display

# In[12]:
# ---- Análise Exploratória de Dados
import seaborn as sns
import matplotlib.pyplot as plt
# heatmap: Amostra de Dados Retangular com Matriz de Codificação por Cores
sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
# corr(): Retorna uma Correlação em um DataFrame
# annot=True: realiza Anotação de Dados em cada Bloco
# cmap: O Mapeamento de Valores de Dados para o Espaço de Cores.
plt.show()
# Pode-se usar:
# - sns.pairplot(tabela)
# - plt.show()

# In[13]:
# ---- Importação de Biblioteca de Machine Learning
from sklearn.model_selection import train_test_split
# Coluna Vendas fica no Eixo y
# Coluna Vendas é Apagada no Eixo x
y = tabela["Vendas"]
x = tabela.drop("Vendas", axis=1)
# train_test_split: Divide Matrizes em Subconjuntos Aleatórios de Treinamento e Teste.
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)
# test_size em Float: Proporção do Conjunto de Dados a ser incluído na Divisão de Teste
# test_size em Int: Representação do Número Absoluto de Amostras de Teste.
# random_state: Controla o Embaralhamento dos Dados antes da Divisão.

# In[14]:
# Realização de Técnicas de Resoluções de Regressão
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# Estruturação da Regressão Linear e da Árvore de Decisão
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()
# Treinamento da Regressão Linear e da Árvore de Decisão
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)
# Bases de Dados de Apoio: x_treino, y_treino, que são ambas train_test_split

# In[15]:
# ---- Teste de Inteligência Artificial e Análise do R²
from sklearn import metrics
# predict: Formulação de Previsões com Teste de AI
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
# r2_score: Comparação da Precisão entre Modelos com R²,
# Parâmetro que diz a Porcentagem de Acerto do Modelo em explicar o que ocorre
print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))  

# In[16]:
# ---- Visualização Gráfica de Previsões
tabela_auxiliar = pd.DataFrame()
# Atribuição de Variáveis
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsões da Árvore de Decisão"] = previsao_arvoredecisao
tabela_auxiliar["Previsões de Regressão Linear"] = previsao_regressaolinear
# Projeção de Gráfico de Linha e Tamanho de Figura
plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()

# In[17]:
# ---- Realização de Previsão Genérica
nova_tabela = pd.read_csv(r"Endereço da Planilha novos.csv")
display(nova_tabela)
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)

# In[18]:
# ---- Projeção de Gráfico de Barras para Previsão
sns.barplot(x=x_treino.columns, y=modelo_arvoredecisao.feature_importances_)
plt.show()