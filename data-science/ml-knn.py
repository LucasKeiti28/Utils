# Carregando as bibliotecas e pacotes
import numpy as np
import pandas as pd 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Carregando o dataset
digitos = datasets.load_digits()
digitos

# Visualizando algumas imagens e labels
images_e_labels = list(zip(digitos.images, digitos.target))
for index, (image, label) in enumerate(images_e_labels[:8]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image, cmap = plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Label:{label}')

# Gerando valores de X e Y
X = digitos.data
Y = digitos.target
print(X.shape, Y.shape)

# Preprocessamento e Normalizacao
# Gerando dados de treino e teste
X_treino, testeData, Y_treino, testeLabels = train_test_split(X, Y, test_size = 0.3, random_state=101)

# Divisao dos dados de treino em dados de validacao (10%)
treinoData, validData, treinoLabels, validLabels = train_test_split(X_treino, Y_treino, test_size = 0.1, random_state = 84)

# Imprimindo o numero de exemplos (observacoes) de cada dataset
print(f"Dados de Treino: {len(treinoData)} observacoes")
print(f"Dados de Validacao: {len(validData)} observacoes")
print(f"Dados de Teste: {len(testeData)} observacoes")

# Normalizacao dos dados pela media
# Calculo da media do dataset
X_norm = np.mean(X, axis = 0)

# Normalizacao dos dados de treino e teste
X_treino_norm = treinoData - X_norm
X_valid_norm = validData - X_norm
X_teste_norm = testeData - X_norm

#Shape dos datasets
print(X_treino_norm.shape, X_valid_norm.shape, X_teste_norm.shape)

# Testando os melhores valores de K
# Range de valores de K que iremos testar
kVals = range(1,30,2)

# Acuracias
acuracias = []

# Loop em todos os valores de K para testar cada um deles
for k in kVals:
    #Treinando o modelo KNN com cada valor de K
    modeloKNN = KNeighborsClassifier(n_neighbors=k)
    modeloKNN.fit(treinoData, treinoLabels)
    #Avaliando o modelo e atualizando a lista de acuracias
    score = modeloKNN.score(validData, validLabels)
    print("Com valor de k = %d, a acuracia e = %.2f%%" % (k, score * 100))
    acuracias.append(score)

#O valor de K que apresentou maior acuracia foi:
i = np.argmax(acuracias)
print("O valor de k = %d alcancou a mais alta acuracia de %.2f%% nos dados de validacao" % (
    kVals[i], acuracias[i] * 100
))

# Construindo e treinando o modelo, considerando o melhor valor de K
modeloFinal = KNeighborsClassifier(n_neighbors=kVals[i])

# Treinando o Modelo
modeloFinal.fit(treinoData, treinoLabels)

# Previsao dos dados de teste e avaliacao do modelo
# Previsao com os dados de teste
predictions = modeloFinal.predict(testeData)

# Performance do Modelo com os dados de teste
print("Avaliacao do modelo, valor de:")
print(classification_report(testeLabels, predictions))

# Confusion matrix do modelo final
print("Confusion Matrix")
print(confusion_matrix(testeLabels, predictions))

# Fazendo previsões com o modelo treinado usando dados de teste
for i in np.random.randint(0, high=len(testeLabels), size=(5,)):
         
    # Obtém uma imagem e faz a previsão
    image = testeData[i]
    prediction = modeloFinal.predict([image])[0]
         
    # Mostra as previsões
    imgdata = np.array(image, dtype='float')
    pixels = imgdata.reshape((8,8))
    plt.imshow(pixels,cmap='gray')
    plt.annotate(prediction,(3,3),bbox={'facecolor':'white'},fontsize=16)
    print("Eu acredito que esse dígito seja: {}".format(prediction))
    plt.show()

# Prevendo digitos com novos dados de entrada
# Definindo um novo dígito (dados de entrada)
novoDigito = [0.,  0.,  0.,  8., 15.,  1.,  0.,  0.,  0.,  0.,  0., 12., 14.,
              0.,  0.,  0.,  0.,  0.,  3., 16.,  7.,  0.,  0.,  0.,  0.,  0.,
              6., 16.,  2.,  0.,  0.,  0.,  0.,  0.,  7., 16., 16., 13.,  5.,
              0.,  0.,  0., 15., 16.,  9.,  9., 14.,  0.,  0.,  0.,  3., 14.,
              9.,  2., 16.,  2.,  0.,  0.,  0.,  7., 15., 16., 11.,  0.]

# Normalizando os dados
novoDigito_norm = novoDigito - X_norm

# Fazendo a previsao com o modelo treinado
novaPrevisao = modeloFinal.predict([novoDigito_norm])

# Previsão do modelo
imgdata = np.array(novoDigito, dtype='float')
pixels = imgdata.reshape((8,8))
plt.imshow(pixels, cmap='gray')
plt.annotate(novaPrevisao,(3,3), bbox={'facecolor':'white'},fontsize=16)
print("Eu acredito que esse dígito seja: {}".format(novaPrevisao))
plt.show()