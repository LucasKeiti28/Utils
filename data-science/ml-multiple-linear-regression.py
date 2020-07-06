# Importing Libs and Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline 

# Carregando o dataset
boston = load_boston()
dataset = pd.DataFrame(boston.data, columns = boston.feature_names)
dataset['target'] = boston.target

# Gerando o numero de observacoes e variaveis
observations = len(dataset)
variables = dataset.columns[:-1]
print(observations, variables)

# Coletando X e y:
X = dataset.iloc[:,:-1]
y = dataset['target'].values

# Adicionando uma coluna de valores 1
Xc = sm.add_constant(X)

# Criando e treinando o modelo
modelo = sm.OLS(y, Xc)
modelo_v1 = modelo.fit()

# Interpretacao do Modelo
modelo_v1.summary()

# Gerando a matriz de Correlacao
X = dataset.iloc[:,:-1]
matriz_corr = X.corr()
print(matriz_corr)

# Criando um Correlation Plot
def visualize_correlation_matrix(data, hurdle = 0.0):
    R = np.corrcoef(data, rowvar = 0)
    R[np.where(np.abs(R) < hurdle)] = 0.0
    heatmap = plt.pcolor(R, cmap = mpl.cm.coolwarm, alpha = 0.8)
    heatmap.axes.set_frame_on(False)
    heatmap.axes.set_yticks(np.arange(R.shape[0]) + 0.5, minor = False)
    heatmap.axes.set_xticks(np.arange(R.shape[1]) + 0.5, minor = False)
    heatmap.axes.set_xticklabels(variables, minor = False)
    plt.xticks(rotation=90)
    heatmap.axes.set_yticklabels(variables, minor = False)
    plt.tick_params(axis = 'both', which = 'both', bottom = 'off', top = 'off', left = 'off', right = 'off') 
    plt.colorbar()
    plt.show()

# Visualizando o Plot
visualize_correlation_matrix(X, hurdle = 0.5)

