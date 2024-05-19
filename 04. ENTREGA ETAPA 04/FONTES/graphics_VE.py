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
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o DataFrame dos dados de veículos elétricos
file_path = "C:\\TEMP\\Electric_Vehicle_Population_Data.csv"
dados = pd.read_csv(file_path)

# Definindo as colunas numéricas para a análise
colunas_numericas = ['Model Year', 'Electric Range', 'Base MSRP']

# Loop para gerar histogramas e box plots para cada coluna numérica
for coluna in colunas_numericas:
    # Histograma
    plt.figure(figsize=(10, 6))
    sns.histplot(dados[coluna].dropna(), bins=30, kde=True)  # Gera histograma com densidade KDE
    plt.title(f'Histograma da coluna {coluna}')  # Adiciona um título ao histograma
    plt.xlabel(coluna)  # Rótulo do eixo X
    plt.ylabel('Frequência')  # Rótulo do eixo Y
    plt.legend(labels=['Densidade KDE', 'Histograma'])  # Adiciona uma legenda
    plt.savefig(f'C:\\TEMP\\{coluna}_histogram.png', dpi=300)  # Salva o histograma como .png
    plt.close()  # Fecha a figura atual

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=dados[coluna].dropna())  # Gera box plot usando Seaborn
    plt.title(f'Box plot da coluna {coluna}')  # Adiciona um título ao box plot
    plt.savefig(f'C:\\TEMP\\{coluna}_boxplot.png', dpi=300)  # Salva o box plot como .png
    plt.close()  # Fecha a figura atual
