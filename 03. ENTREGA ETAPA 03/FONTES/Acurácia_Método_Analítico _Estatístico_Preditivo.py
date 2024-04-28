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


# 3. Extraindo uma amostra da população para aplicar técnicas estatisticas.

eletric_range_df = df[(df['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)') & 
                      (df['Electric Range']  != 0) & (df['Model Year'].isin(np.arange(2018,2024,1)))][['Electric Range']]

# 3.1. Calculando média, variância e desvio padrão
eletric_range_media_df = eletric_range_df.mean()
eletric_range_variancia_df = eletric_range_df.var()
desvio_padrao = np.sqrt(eletric_range_variancia_df)


# 3.2. Realizando fit dos dados para uma função log normal
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimativa dos parÂmetros para fazer o fit
    for ii, distribution in enumerate([d for d in _distn_names if d in ['lognorm']]):

        print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution = getattr(stats, distribution)

        # Tentar fazer o fit na distribuição
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                params = distribution.fit(data)

                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculando a função densidade de probabilidade e a CDF (Cumulative Density Function)
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                cdf = distribution.cdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                best_distributions.append((distribution, params, sse, cdf))
        
        except Exception:
            pass

    
    return sorted(best_distributions, key=lambda x:x[2])


# 3.3. Plotando o histogranma para o electric range, sua função densidade de 
# probabilidade e a aproximação da mesma para uma log normal
ax = sns.distplot(eletric_range_df, hist=True)

best_distributions = best_fit_distribution(eletric_range_df, 200, ax)
best_dist = best_distributions[0]

sns.set_style("whitegrid")
plt.title('Histograma Eletric Range', fontsize=16)
plt.xlabel('Eletric Range', fontsize=12)
plt.ylabel('Freq', fontsize=12)
plt.show()


plt.plot(best_dist[3])
plt.show()
