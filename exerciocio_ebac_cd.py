import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregamento de Dados
arquivo = 'exercicios_visu_dados\ecommerce_preparados.csv'
dados = pd.read_csv(arquivo)

# Inspeção Inicial
print('Tipos das colunas:\n')
print(dados.dtypes)

print('\nValores ausentes por coluna:\n')
print(dados.isnull().sum())

# Tratamento de Dados Ausentes
colunas_texto = ['Material', 'Gênero']

for coluna in colunas_texto:
    dados[coluna] = dados[coluna].fillna('Não Informado')

dados.fillna(0, inplace=True)

print('\nTotal de valores nulos após tratamento:', dados.isnull().sum().sum())

# Padronização de textos
colunas_padrao = ['Marca', 'Material', 'Temporada']

for col in colunas_padrao:
    dados[col] = dados[col].astype(str).str.title()

# Normalização das estações do ano
padrao_estacoes = {
    'Não Definido': 'Indefinido',
    'Outono/Inverno': 'Outono-Inverno',
    'Outono-Inverno': 'Outono-Inverno',
    'Primavera/Verão': 'Primavera-Verão',
    'Primareva-Verão': 'Primavera-Verão',
    'Primavera/Verão/Outono/Inverno': 'Todas as Estações',
    'Primavera/Verão Outono/Inverno': 'Todas as Estações',
    'Primavera-Verão Outono-Inverno': 'Todas as Estações',
    'Primavera-Verão - Outono-Inverno': 'Todas as Estações',
}

dados['Temporada'] = dados['Temporada'].replace(padrao_estacoes)

# Remoção de Dados Inválidos
dados = dados[dados['Temporada'] != '2021']

# Visualização das Amostras
print('\nPrimeiras Linhas:\n')
print(dados.head(10))

print('\nÚltimas Linhas:\n')
print(dados.tail(10))

# Gráfico de Barras
contagem_temporadas = dados['Temporada'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(contagem_temporadas.index, contagem_temporadas.values)
plt.title('Quantidade de Vendas por Temporada')
plt.xlabel('Temporada')
plt.ylabel('Total de Vendas')
plt.show()

# Gráfico de Pizza
plt.figure(figsize=(14, 8))
plt.pie(
    contagem_temporadas.values,
    labels=contagem_temporadas.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title('Participação das Vendas por Estação')
plt.show()

# Pairplot (Análise Multivariada)
variaveis_numericas = ['Qtd_Vendidos_Cod', 'Preço', 'Marca_Cod']
sns.pairplot(dados[variaveis_numericas])
plt.show

# Curva de Dansidade de Vendas
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=dados,
    x='Qtd_Vendidos_Cod',
    fill=True
)
plt.title('Distribuição de Densidade das Vendas')
plt.xlabel('Quantidade Vendida')
plt.show()

# Gráfico de Regressão
plt.figure(figsize=(8, 6))
sns.regplot(
    data=dados,
    x='Qtd_Vendidos_Cod',
    y='N_Avaliações',
    scatter_kws={'alpha': 0.5}
)
plt.title('Relação entre Vendas e Avaliações')
plt.xlabel('Quantidade Vendida')
plt.ylabel('Número de Avaliações')
plt.show()

# Mapa de Calor de Correlação
matriz_correlacao = dados[['Qtd_Vendidos_Cod', 'N_Avaliações']].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(
    matriz_correlacao,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title('Correlação entre Vendas e Avaliações')
plt.show()