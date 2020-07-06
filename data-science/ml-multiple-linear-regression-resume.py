## Resumo: Fazendo previsao com modelo de regressao linear multipla

# importando bibliotecas
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Carregando o dataset
boston = load_boston()
dataset = pd.DataFrame(boston.data, columns=boston.feature_names)
dataset['target'] = boston.target

# Formato do dataset
dataset.shape
dataset.head()

# Coletando os valores de X e y:
# Usaremos como variaveis explanatorias, apenas as 4 mais relevantes
X = dataset[['LSTAT', 'RM', 'DIS', 'PTRATIO']]
y = dataset['target'].values

# Dividindo entre dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

# Criando o modelo
modelo = LinearRegression(normalize=False, fit_intercept=True)

# Treinando o modelo
modelo_v2 = modelo.fit(X_train, y_train)

# Calculando o R squared do modelo
r2_score(y_test, modelo_v2.fit(X_train, y_train).predict(X_test))

# Produz a matriz com os novos dados de entrada para a previsão
LSTAT = 5
RM = 8
DIS = 6
PTRATIO = 19

# Lista com os valores das variáveis
dados_nova_casa = [LSTAT, RM, DIS, PTRATIO]

# Reshape
Xp = np.array(dados_nova_casa).reshape(1, -1)

# Previsão
print("Taxa Média de Ocupação Para a Casa:", modelo_v2.predict(Xp))