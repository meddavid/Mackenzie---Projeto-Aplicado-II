import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
import seaborn as sns
import fsspec
import numpy as np
import seaborn as sns

base_dados = pd.read_csv('C:\Users\avsan\.vscode\Projeto Aplicado II\alt_fuel_stations (Mar 18 2024).csv')
base_dados.head()
base_dados.info()
base_dados.dropna()

base_dados.isnull().sum() / len(base_dados)

print('Before', len(base_dados))
base_dados = base_dados.dropna()
print('After', len(base_dados))

colunas_selecionadas = ['City', 'State', 'Model Year', 'Make', 'Model',
                        'Electric Vehicle Type', 'Electric Range',
                        'Electric Utility', 'Vehicle Location']

dados_selecionados = base_dados[colunas_selecionadas]

# Criando um DataFrame com os dados selecionados
dados_selecionados = pd.DataFrame(dados_selecionados)
df = dados_selecionados.copy()

df.head(15)
df.tail()
df.index
df.columns
df.shape


# Separar os dados por marcas
marcas = df['Make'].unique()

# Dicionário para armazenar os subdataframes
subdataframes = {}

# Iterando sobre as marcas únicas
for marca in marcas:
    # Criando o subdataframe para cada marca
    subdataframes[marca] = df[df['Make'] == marca]

# Consultando e criando um novo dataset com as primeiras linhas do subdataframe por marca
for marca_desejada in subdataframes.keys():
    print(f"\nSubdataframe da marca {marca_desejada}:")
    sub_df = subdataframes[marca_desejada] #aqui pode colocar a consulta, ex .head()
    globals()[f"sub_df_{marca_desejada}"] = sub_df
    print(sub_df)
    
# Defina o número de gráficos por página
graficos_por_pagina = 6
linhas_por_pagina = 2
colunas_por_pagina = 3

# Tamanho da fonte do título do gráfico
tamanho_fonte_titulo = 8

# Inicialize uma variável para contar os gráficos
contador_graficos = 0

# Inicialize uma lista vazia para armazenar os eixos
eixos = None

# Iterando sobre os subdatasets para gerar gráficos
for marca, sub_df in subdataframes.items():
    # Contando a frequência dos modelos de carros em cada marca
    contagem_modelos = sub_df['Model'].value_counts()
    
    # Verificando se é necessário criar uma nova página
    if contador_graficos % graficos_por_pagina == 0:
        # Criando uma nova figura e eixos
        fig, eixos = plt.subplots(linhas_por_pagina, colunas_por_pagina, figsize=(15, 10))
    
    # Calculando a posição do gráfico na página
    linha = contador_graficos // colunas_por_pagina % linhas_por_pagina
    coluna = contador_graficos % colunas_por_pagina
    
    # Criando o gráfico de barras no subplot correspondente
    contagem_modelos.plot(kind='bar', color='#1f77b4', edgecolor='black', linewidth=3, ax=eixos[linha, coluna])
    eixos[linha, coluna].set_title(f'Frequência dos modelos de carros da marca {marca}', fontsize=tamanho_fonte_titulo)
    eixos[linha, coluna].set_xlabel('Modelo')
    eixos[linha, coluna].set_ylabel('Frequência')
    eixos[linha, coluna].tick_params(axis='x', rotation=45)
    
    # Incrementando o contador de gráficos
    contador_graficos += 1

# Ajustando o layout da última página de gráficos
plt.tight_layout()
# Exibindo a última página de gráficos
plt.show()

# Iterando sobre os subdatasets para contar o número de BEVs e PHEVs
for marca, sub_df in subdataframes.items():
    # Contando o número de BEVs e PHEVs em cada marca
    contagem_ev_type = sub_df['Electric Vehicle Type'].value_counts()
    
    # Exibindo as contagens de BEVs e PHEVs
    print(f"\nMarca: {marca}")
    print("Contagem de Electric Vehicle Type:")
    print(contagem_ev_type)

# Defina o número de gráficos por página
graficos_por_pagina = 6

# Tamanho da fonte do título do gráfico
tamanho_fonte_titulo = 8

# Calculando o número de linhas e colunas para os gráficos
num_linhas, num_colunas = 2, 3
graf_por_pagina_por_area = graficos_por_pagina // (num_linhas * num_colunas)

# Inicialize uma variável para contar os gráficos
contador_graficos = 0

# Inicialize uma variável para contar as páginas
contador_paginas = 0

# Criando uma nova figura e eixos fora do loop de iteração
fig, axs = plt.subplots(num_linhas, num_colunas, figsize=(15, 10))

# Iterando sobre os subdatasets para contar o número de BEVs e PHEVs e gerar gráficos
for marca, sub_df in subdataframes.items():
    # Contando o número de BEVs e PHEVs em cada marca
    contagem_ev_type = sub_df['Electric Vehicle Type'].value_counts()
    
    # Criando o gráfico de barras
    linha = contador_graficos % num_linhas
    coluna = contador_graficos % num_colunas
    ax = axs[linha, coluna]
    contagem_ev_type.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=3, ax=ax)
    ax.set_title(f'Contagem de Electric Vehicle Type para a marca {marca}', fontsize=tamanho_fonte_titulo)
    ax.set_xlabel('Tipo de Veículo Elétrico')
    ax.set_ylabel('Contagem')
    ax.tick_params(axis='x', rotation=0)
    
    # Incrementando o contador de gráficos
    contador_graficos += 1

    # Verificando se é necessário criar uma nova página
    if contador_graficos % graf_por_pagina_por_area == 0:
        # Exibindo a página atual de gráficos
        plt.show()
        # Incrementando o contador de páginas
        contador_paginas += 1
        # Criando uma nova figura e eixos para a próxima página
        fig, axs = plt.subplots(num_linhas, num_colunas, figsize=(15, 10))

# Ajustando o layout da última página de gráficos
plt.tight_layout()
# Exibindo a última página de gráficos
plt.show()



# Defina o número de gráficos por página
graficos_por_pagina = 6

# Tamanho da fonte do título do gráfico
tamanho_fonte_titulo = 8

# Dicionário para armazenar o número de veículos BEV por marca
num_bev_por_marca = {}

# Iterar sobre os subdatasets originais
for marca, sub_df in subdataframes.items():
    # Filtrar o subdataset para selecionar apenas os carros do tipo BEV
    sub_df_bev = sub_df[sub_df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']

    # Armazenar o número de veículos BEV por marca
    num_bev = len(sub_df_bev)

    # Adicionar a marca e o número de veículos ao dicionário
    num_bev_por_marca[marca] = num_bev

# Ordenar o dicionário num_bev_por_marca em ordem decrescente por valor
num_bev_por_marca_ordenado = sorted(num_bev_por_marca.items(), key=lambda x: x[1], reverse=True)

# Dividir o dicionário ordenado em lotes de tamanho graficos_por_pagina
lotes = [num_bev_por_marca_ordenado[i:i + graficos_por_pagina] for i in range(0, len(num_bev_por_marca_ordenado), graficos_por_pagina)]

# Paleta de cores dinâmica com seaborn
cores = sns.color_palette('husl', len(lotes[0]))

# Criar e exibir os gráficos
for lote in lotes:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (marca, num_bev) in enumerate(lote):
        ax.bar(marca, num_bev, color=cores[i], edgecolor='black', linewidth=3)
    ax.set_title('Número de veículos BEV por marca', fontsize=tamanho_fonte_titulo)
    ax.set_xlabel('Marca')
    ax.set_ylabel('Número de veículos BEV')
    ax.set_xticklabels([marca for marca, _ in lote], rotation=45)
    plt.tight_layout()
    plt.show()

# Filtrar o dataset original para selecionar apenas os carros do tipo BEV
df_bev = df.loc[df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']

soma_quantidade_total = df_bev['Model'].value_counts()
total = soma_quantidade_total.sum()
print(total)
def consultar_veiculos_bev(df, marcas_de_interesse=None, modelos_de_interesse=None):
   
    print("Tamanho do DataFrame original:", len(df))  # Depuração: Verificar o tamanho do DataFrame original

    # Filtrar o dataset original para selecionar apenas os carros do tipo BEV
    df_bev = df[df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']

    print("Tamanho do DataFrame filtrado (BEV):", len(df_bev))  # Depuração: Verificar o tamanho do DataFrame filtrado

    # Filtrar por marcas de interesse (se especificado)
    if marcas_de_interesse:
        df_bev = df_bev[df_bev['Make'].isin(marcas_de_interesse)]

    print("Tamanho do DataFrame filtrado (marcas de interesse):", len(df_bev))  # Depuração: Verificar o tamanho do DataFrame filtrado

    # Filtrar por modelos de interesse (se especificado)
    if modelos_de_interesse:
        df_bev = df_bev[df_bev['Model'].isin(modelos_de_interesse)]

    print("Tamanho do DataFrame filtrado (modelos de interesse):", len(df_bev))  # Depuração: Verificar o tamanho do DataFrame filtrado

    # Agrupar os veículos BEV por marca e modelo, contar a quantidade e manter as colunas de interesse
    df_bev_por_marca_modelo = df_bev.groupby(['Make', 'Model']).agg(
        Quantidade=('Model', 'size'),  # Conta o número de linhas para cada grupo (ou seja, a quantidade de veículos)
        Electric_Range=('Electric_Range', 'first'),  # Pega a autonomia elétrica do primeiro veículo do grupo
        Vehicle_Location=('Vehicle Location', 'first')  # Pega a localização do primeiro veículo do grupo
    ).reset_index()

    print("Tamanho do DataFrame após a agregação:", len(df_bev_por_marca_modelo))  # Depuração: Verificar o tamanho do DataFrame após a agregação

    # Reordenar as colunas
    df_bev_por_marca_modelo_ordenado = df_bev_por_marca_modelo.sort_values(by='Quantidade', ascending=False)

    return df_bev_por_marca_modelo_ordenado



df = pd.read_csv('D:/OneDrive - Instituto Presbiteriano Mackenzie/Mackenzie/Aulas/3 semestre/Projeto Aplicado II/alt_fuel_stations (Mar 18 2024).csv')

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

colunas_gps = colunas_selecionadas_preenchidas[['Latitude', 'Longitude']]

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



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados
base_dados = pd.read_csv('D:/OneDrive - Instituto Presbiteriano Mackenzie/Mackenzie/Aulas/3 semestre/Projeto Aplicado II/Electric_Vehicle_Population_Data.csv', index_col=False)

# Filtrar os dados para incluir apenas veículos desde 2015 e do tipo BEV
base = base_dados[(base_dados['Model Year'] >= 2015) & (base_dados['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)')]

# Agrupar por ano e calcular a quantidade total de veículos por ano
base_agrupada = base.groupby('Model Year').size().reset_index(name='Quantidade')

# Pré-processamento de dados
X = base_agrupada["Model Year"].values.reshape(-1, 1)
y = base_agrupada["Quantidade"].values

# Dividir dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliar o modelo
print("R²:", model.score(X_test, y_test))
print("RMSE:", np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

# Fazer previsões para um futuro "Model Year"
ano_futuro = 2030
previsao = model.predict([[ano_futuro]])

print(f"Previsão de vendas para {ano_futuro}: {previsao}")

# Criar um vetor de "Model Year" para as previsões
anos_futuro = np.arange(2023, 2030)

# Fazer previsões para os "Model Years" futuros
previsoes = model.predict(anos_futuro.reshape(-1, 1))

# Plotar os dados
plt.scatter(X_test, y_test, color="blue", label="Dados")
plt.plot(X_test, model.predict(X_test), color="red", label="Linha de regressão")
plt.plot(anos_futuro, previsoes, color="green", label="Previsões")

# Ajustar o gráfico
plt.legend()
plt.xlabel("Model Year")
plt.ylabel("Quantidade")
plt.title("Regressão Linear")
plt.show()



# Importar as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Dados de modelo linear ajustado previamente
lm = LinearRegression()
lm.fit(X_train, y_train)

# Cenário otimista: crescimento de 10% ao ano
optimistic_growth_rate = 0.10
ano_otimista = 2030
crescimento_estimado_otimista = model.predict([[ano_otimista]])[0] * (1 + optimistic_growth_rate)
print("Cenário otimista - Quantidade estimada de veículos em 2030:", crescimento_estimado_otimista)

# Cenário pessimista: crescimento de 5% ao ano
pessimistic_growth_rate = 0.05
ano_pessimista = 2030
crescimento_estimado_pessimista = model.predict([[ano_pessimista]])[0] * (1 + pessimistic_growth_rate)
print("Cenário pessimista - Quantidade estimada de veículos em 2030:", crescimento_estimado_pessimista)

# Cenário realista: crescimento de 7% ao ano
realistic_growth_rate = 0.07
ano_realista = 2030
crescimento_estimado_realista = model.predict([[ano_realista]])[0] * (1 + realistic_growth_rate)
print("Cenário realista - Quantidade estimada de veículos em 2030:", crescimento_estimado_realista)

















