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
# %matplotlib inline 

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

# Processo para estudo da multicolinearidade das variaveis: eigevectors e eigevalues.
# Gerando eigenvalues e eigenvectors
corr = np.corrcoef(X, rowvar = 0)
eigenvalues, eigenvectors = np.linalg.eig(corr)

# Procurar pelo menor valor em eigenvalues, que representa grande chance de multicolinearidade
print(eigenvalues)

# Repare que em valores absoluto os index de posicao 2, 8 e 9 se destacam
print(eigenvectors[:,8])

# Imprimindo os nomes das variaveis para saber quais contribuem mais com seus valores para construir o autovetor.
# Recomenda-se tratar os valores dessa coluna ou ate mesmo deletar.
print(variables[2], variables[8], variables[9])

# Gradiente Descendente
# Feature Scaling
# Aplicando Padronização
standardization = StandardScaler()
Xst = standardization.fit_transform(X)
original_means = standardization.mean_
originanal_stds = standardization.scale_

# Gerando X e Y
Xst = np.column_stack((Xst,np.ones(observations)))
y  = dataset['target'].values

import random
import numpy as np

def random_w( p ):
    return np.array([np.random.normal() for j in range(p)])

def hypothesis(X,w):
    return np.dot(X,w)

def loss(X,w,y):
    return hypothesis(X,w) - y

def squared_loss(X,w,y):
    return loss(X,w,y)**2

def gradient(X,w,y):
    gradients = list()
    n = float(len( y ))
    for j in range(len(w)):
        gradients.append(np.sum(loss(X,w,y) * X[:,j]) / n)
    return gradients

def update(X,w,y, alpha = 0.01):
    return [t - alpha*g for t, g in zip(w, gradient(X,w,y))]

def optimize(X,y, alpha = 0.01, eta = 10**-12, iterations = 1000):
    w = random_w(X.shape[1])
    path = list()
    for k in range(iterations):
        SSL = np.sum(squared_loss(X,w,y))
        new_w = update(X,w,y, alpha = alpha)
        new_SSL = np.sum(squared_loss(X,new_w,y))
        w = new_w
        if k>=5 and (new_SSL - SSL <= eta and new_SSL - SSL >= -eta):
            path.append(new_SSL)
            return w, path
        if k % (iterations / 20) == 0:
            path.append(new_SSL)
    return w, path               

# Imprimindo o resultado                           
alpha = 0.01
w, path = optimize(Xst, y, alpha, eta = 10**-12, iterations = 20000)
print ("Coeficientes finais padronizados: " + ', '.join(map(lambda x: "%0.4f" % x, w)))          

# Desfazendo a Padronização
unstandardized_betas = w[:-1] / originanal_stds
unstandardized_bias  = w[-1]-np.sum((original_means / originanal_stds) * w[:-1])

# Imprimindo o resultado
print ('%8s: %8.4f' % ('bias', unstandardized_bias))
for beta,varname in zip(unstandardized_betas, variables):
    print ('%8s: %8.4f' % (varname, beta))

# Importancia dos Atributos
# Criando um modelo
modelo = linear_model.LinearRegression(normalize=False, fit_intercept=True)

# Treinando o modelo com dados nao padronizados (em escalas diferentes)
modelo.fit(X,y)

# Imprimindo os coeficientes e variaveis
for coef, var in sorted(zip(map(abs, modelo.coef_), dataset.columns[:-1]), reverse = True):
    print("%6.3f %s" % (coef,var))

# Padronizando os dados
standardization = StandardScaler()
Stand_coef_linear_reg = make_pipeline(standardization, modelo)

# Treinando o modelo com dados padronizados (na mesma escala)
Stand_coef_linear_reg.fit(X,y)

# Imprimindo os coeficientes e as variáveis
for coef, var in sorted(zip(map(abs, Stand_coef_linear_reg.steps[1][1].coef_), dataset.columns[:-1]), reverse = True):
    print ("%6.3f %s" % (coef,var))

# Usando o RSquared
modelo = linear_model.LinearRegression(normalize=False, fit_intercept=True)

def r2_est(X,y):
    return r2_score(y, modelo.fit(X,y).predict(X))

print(f"Coeficiente R2: {round(r2_est(X,y),3)}")

# Gera o impacto de cada atributo no R2
r2_impact = list()
for j in range(X.shape[1]):
    selection = [i for i in range(X.shape[1]) if i!=j]
    r2_impact.append(((r2_est(X,y) - r2_est(X.values[:,selection],y)), dataset.columns[j]))
    
for imp, varname in sorted(r2_impact, reverse = True):
    print ('%6.3f %s' %  (imp, varname))

