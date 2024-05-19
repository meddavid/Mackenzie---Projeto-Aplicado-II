=====================================================================================
Programa............: SUMMARY
Autor...............: Natália Françozo
Data................: 26/04/2024
Descrição / Objetivo: Analise exploratória focada no modelo de negócio
Doc. Origem.........: Electric_Vehicle_Population_Data.csv
Solicitante.........: Professor Felipe Cunha
Uso.................: Projeto Apliado II
Modificações........: 26/04/2024 - Desenvolvimento
=====================================================================================

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import warnings
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Importar base de dados

base_dados = pd.read_csv('Electric_Vehicle_Population_Data.csv')

print(base_dados.head())
base_dados.info()

# 2. Preparaçao dos Dados e seleção das colunas a serem usadas e "drop" em dados vazios ou nulos.

base_dados.isnull().sum() / len(base_dados)

print('Before', len(base_dados))


dados_selecionados = pd.DataFrame(base_dados)
colunas_selecionadas = ['City', 'State', 'Model Year', 'Make', 'Model',
                        'Electric Vehicle Type', 'Electric Range',
                        'Electric Utility', 'Vehicle Location'] 


df = dados_selecionados[colunas_selecionadas]

df = df.dropna()
print('After', len(base_dados))

df.head(15)
df.tail()
df.index
df.columns
df.shape

# 1. Agrupa todos os veículo BEV por estado e gera gráfico de barras

estado_BEV_df = df[['State', 'Electric Vehicle Type']]

estado_BEV_df = estado_BEV_df[estado_BEV_df['Electric Vehicle Type']=='Battery Electric Vehicle (BEV)']

estado_BEV_count_df = estado_BEV_df['State'].value_counts()

estado_BEV_count_df.plot(kind='bar',  logy=True, title='BEV por Estado')
plt.show()

# 2. Agrupa todos os veículo PHEV por estado e gera gráfico de barras

marca_PHEV_df = df[['State', 'Electric Vehicle Type']]

marca_PHEV_df = marca_PHEV_df[marca_PHEV_df['Electric Vehicle Type']=='Plug-in Hybrid Electric Vehicle (PHEV)']

estado_PHEV_count_df = marca_PHEV_df['State'].value_counts()

estado_PHEV_count_df.plot(kind='bar',  logy=True, title='PHEV por Estado')
plt.show()

# 3. Ranking dos 10 Cidades com maior quantidade de veículos eletricos em WA

top_10_cidades_df = df[['City', 'State']]

top_10_cidades_df = top_10_cidades_df[top_10_cidades_df['State']=='WA']

top_10_cidades_count_df = top_10_cidades_df['City'].value_counts()

top_10_cidades_df = top_10_cidades_count_df.head(10)

top_10_cidades_df.plot(kind='bar',  logy=True, title='10 Cidades com Mais Veículos Elétricos No Estado WA')
plt.show()

# 4. Modelos de veículos eletricos por ano - Histograma Normalizado.

modelo_ano_df = df['Model Year']

plt.hist(modelo_ano_df, bins=10, log=True, density=True)
plt.title('Veículos Elétricos por Ano (Histograma)')
plt.xlabel('Ano do Modelo')
plt.ylabel('Densidade de Probabilidade')
plt.show()

# 5. Agrupa todos os veículo BEV e fazer a distribuição do alcance elétrico (Eletric Range)

frequencia_BEV_df = df[['Electric Range', 'Electric Vehicle Type']]

frequencia_BEV_df = frequencia_BEV_df[frequencia_BEV_df['Electric Vehicle Type']=='Battery Electric Vehicle (BEV)']

frequencia_BEV_df = frequencia_BEV_df[frequencia_BEV_df['Electric Range'] != 0]

frequencia_BEV_count_df = frequencia_BEV_df['Electric Range'].value_counts()

frequencia_BEV_count_df.plot(kind='bar',  logy=True, title='Autonomia de Veículo BEV')
plt.show()

# 6. Agrupa todos os veículo PHEV e fazer a distribuição do alcance elétrico (Eletric Range)

frequencia_PHEV_df = df[['Electric Range', 'Electric Vehicle Type']]

frequencia_PHEV_df = frequencia_PHEV_df[frequencia_PHEV_df['Electric Vehicle Type']=='Plug-in Hybrid Electric Vehicle (PHEV)']

frequencia_PHEV_df = frequencia_PHEV_df[frequencia_PHEV_df['Electric Range'] != 0]

frequencia_PHEV_count_df = frequencia_PHEV_df['Electric Range'].value_counts()

frequencia_PHEV_count_df.plot(kind='bar',  logy=True, title='Autonomia de Veículo PHEV')
plt.show()

# 7. Verificar se houve aumento de potencia com o passar dos anos

ano_autonomia_df = df[['Model Year', 'Electric Range', 'Make']]

ano_autonomia_df = ano_autonomia_df[ano_autonomia_df['Make'].isin(['TESLA', 'KIA', 'MITSUBISHI', 'NISSAN','VOLVO'])]
ano_autonomia_df = ano_autonomia_df[ano_autonomia_df['Electric Range'] != 0]

ano_autonomia_df.set_index('Model Year', inplace=True)
ano_autonomia_count_df = ano_autonomia_df.groupby(['Model Year', 'Make'])

ano_autonomia_count_df['Electric Range'].mean().unstack().plot(kind='line', marker='o', title='Autonomia de Cada Marca ao Longo dos Anos')
plt.show()

