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

# Caminho atualizado do arquivo CSV
file_path = "C:\\TEMP\\pontos_carregamento.csv"

# Carregando o arquivo CSV
df = pd.read_csv(file_path)

# Exibindo as primeiras linhas para uma visão geral
first_rows = df.head()

# Resumo da estrutura do conjunto de dados
structure = df.info()

# Resumo estatístico básico
summary = df.describe()

first_rows, structure, summary
