'''
=======================================================================
Programa............: Analitics
Autor...............: Grupo Mcgyver
Data................: 15/04/2024
Descrição / Objetivo: Exibição de Análise de Metadados
Doc. Origem.........: Electric_Vehicle_Population_Data.csv
Solicitante.........: Professor Felipe Cunha
Uso.................: Projeto Apliado II
Modificações........: 15/04/2024- Desenvolvimento
=======================================================================
'''
import pandas as pd

# Caminho do arquivo CSV
file_path = "C:\\TEMP\\pontos_carregamento.csv"

# Carregando o arquivo CSV
df = pd.read_csv(file_path)

# Exibindo as primeiras linhas para uma visão geral
print("Primeiras linhas do conjunto de dados:")
print(df.head(), "\n")

# Resumo da estrutura do conjunto de dados
print("Resumo da estrutura do conjunto de dados:")
df.info()

# Resumo estatístico básico das colunas numéricas
print("\nResumo estatístico básico das colunas numéricas:")
colunas_numericas = df.select_dtypes(include=[float, int]).columns.tolist()  # Automatizando a escolha de colunas numéricas
print(df[colunas_numericas].describe(), "\n")

# Analisando a quantidade de registros
num_registros = len(df)
print(f"#### - Quantidade de registros: {num_registros}\n")
