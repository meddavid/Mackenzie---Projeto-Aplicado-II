'''
=====================================================================================
Programa............: SUMMARY
Autor...............: Grupo Mcgyver
Data................: 01/03/2024
Descrição / Objetivo: Exibição de Análise de Metadados
Doc. Origem.........: Electric_Vehicle_Population_Data.csv
Solicitante.........: Professor Felipe Cunha
Uso.................: Projeto Apliado II
Modificações........: 01/03/2024 - Desenvolvimento
=====================================================================================
'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o DataFrame dos pontos de carregamento
file_path = "C:\\TEMP\\pontos_carregamento.csv"  # Caminho local para o arquivo
dados = pd.read_csv(file_path)

# Lista das colunas numéricas que serão analisadas
colunas_numericas = ['Latitude', 'Longitude']

# Loop para gerar histogramas e box plots para cada coluna numérica
for coluna in colunas_numericas:
    # Histograma
    plt.figure(figsize=(10, 6))
    sns.histplot(dados[coluna], bins=30, kde=True)  # Gera histograma com densidade KDE
    plt.title(f'Histograma da coluna {coluna}')  # Adiciona um título ao histograma
    plt.xlabel(coluna)  # Rótulo do eixo X
    plt.ylabel('Frequência')  # Rótulo do eixo Y
    plt.legend(labels=['Densidade KDE', 'Histograma'])  # Adiciona uma legenda
    histogram_path = f'C:\\TEMP\\{coluna}_histogram.png'
    plt.savefig(histogram_path, dpi=300)  # Salva o histograma como .png com alta resolução
    plt.close()  # Fecha a figura atual

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=dados[coluna])  # Gera box plot usando Seaborn
    plt.title(f'Box plot da coluna {coluna}')  # Adiciona um título ao box plot
    plt.xlabel(coluna)  # Rótulo do eixo X
    boxplot_path = f'C:\\TEMP\\{coluna}_boxplot.png'
    plt.savefig(boxplot_path, dpi=300)  # Salva o box plot como .png com alta resolução
    plt.close()  # Fecha a figura atual
