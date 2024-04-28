=====================================================================================
Programa............: SUMMARY
Autor...............: FELIPE JOSÉ DA CUNHA e ANA VITÓRIA SILVA
Data................: 26/04/2024
Descrição / Objetivo: Exibição de Análise de Metadados
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

# Inserindo um novo dataset "pontos_carregamento.csv"

df = pd.read_csv('pontos_carregamento.csv')

colunas_selecionadas = df.copy()

colunas_selecionadas = colunas_selecionadas[['State','Groups With Access Code','EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count', 'Latitude', 'Longitude']]


# Crie um novo DataFrame contendo apenas as linhas onde a coluna 'category' é igual a 'public'
colunas_selecionadas_public = colunas_selecionadas[colunas_selecionadas['Groups With Access Code'] == 'Public']


# Crie um novo DataFrame contendo apenas as linhas onde a coluna 'State' é igual a 'WA'
colunas_selecionadas_public_wa = colunas_selecionadas_public[colunas_selecionadas_public['State'] == 'WA']

# Crie uma cópia do DataFrame original para manter as outras colunas inalteradas
df_filtered = colunas_selecionadas_public_wa.copy()

# Selecione apenas as colunas específicas para verificar se todas estão preenchidas com NaN
columns_to_check = ['EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count']

# Remova as linhas onde todas as três colunas especificadas estão preenchidas com NaN
df_filtered = df_filtered.dropna(subset=columns_to_check, how='all')


df_filtered[['EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count']] = df_filtered[['EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count']].fillna(0)


# Selecione apenas as colunas desejadas para calcular a soma
colunas_selecionadas_soma = ['EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count']

# Calcule a soma das instâncias das colunas selecionadas
soma_instancias = df_filtered[colunas_selecionadas_soma].sum()

# Crie uma tabela com o resultado da soma
tabela_soma = pd.DataFrame({'Colunas': colunas_selecionadas_soma, 'Soma das Instâncias': soma_instancias})

# Crie o gráfico de barras coloridas com base nas somas das instâncias
plt.figure(figsize=(10, 6))
soma_instancias.plot(kind='bar', color=['blue', 'green', 'red'])
plt.title('Soma das Instâncias dos Pontos de Recarga por Capacidade')
plt.xlabel('Colunas')
plt.ylabel('Soma das Instâncias')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Exiba a tabela com o resultado da soma
print("Tabela com o resultado da soma das instâncias das colunas selecionadas:")
print(tabela_soma)

colunas_gps = colunas_selecionadas_public[['Latitude', 'Longitude']]

import pandas as pd
import re

# Função para extrair latitude e longitude de uma string no formato POINT (longitude latitude)
def extrair_latitude_longitude(point_string):
    # Use uma expressão regular para extrair os valores numéricos
    matches = re.findall(r"-?\d+\.\d+", point_string)
    if len(matches) >= 2:
        return float(matches[1]), float(matches[0])  # A ordem é invertida para corresponder ao formato (latitude, longitude)
    else:
        return None, None

seu_dataframe = base_dados
# Iterar sobre as linhas do dataframe e extrair latitude e longitude
dados_selecionados = []
for index, linha in seu_dataframe.iterrows():
    localizacao_str = linha['Vehicle Location']
    latitude, longitude = extrair_latitude_longitude(localizacao_str)
    if latitude is not None and longitude is not None:
        dados_selecionados.append({'Latitude': latitude, 'Longitude': longitude})

# Imprimir os dados
for dado in dados_selecionados:
    print(dado)
    
gps_selecionado_carros = dado

#Precisamos calcular a densidade de veiculos por unidade
#por area: Formula Densidade = Numero de veiculos/Area da Regiao em quilometros quadrados
#area da regiao de Washingtom em km2 177 quilometros segundo site https://www.greelane.com/pt/humanidades/geografia/washington-dc-geography-1435747/#:~:text=DC%20tem%2068%20milhas%20quadradas%20A%20%C3%A1rea%20total,m%29%20e%20est%C3%A1%20localizado%20no%20bairro%20de%20Tenleytown.
Formula_Densidade = 135364 / 177
print(Formula_Densidade)

#Densidade de veiculos em relação a capacidade de carga.
#levar em consideração a capacidade de carga dos pontos em relação
#aos diferentes tipos de pontos com capacidade variada
#formula densidade = numero de veiculos / capacidade total dos pontos de recarga
capacidade_EV_Level1 = 135364 / 17
print(capacidade_EV_Level1)


capacidade_EV_Level2= 135364 / 1045
print(capacidade_EV_Level2)
   
EV_DC_Fast_Count = 135364 / 4175
print(EV_DC_Fast_Count)