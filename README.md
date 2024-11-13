## Descrição
Este projeto implementa uma rede neural convolucional (CNN) simples para classificar dígitos manuscritos utilizando o conjunto de dados MNIST. O MNIST é um famoso benchmark para a avaliação de algoritmos de aprendizado de máquina, consistindo em 70.000 imagens de dígitos manuscritos (0-9) em escala de cinza.

## Tecnologias Utilizadas
Python: Linguagem de programação utilizada para a implementação do modelo.
TensorFlow: Biblioteca de código aberto para aprendizado de máquina, utilizada para construir e treinar modelos de redes neurais.
Keras: API de alto nível integrada ao TensorFlow que facilita a criação e o treinamento de modelos de aprendizado profundo.
Matplotlib: Biblioteca de visualização de dados que é usada para exibir imagens e resultados.

## Estrutura do Código
#### Importação das Bibliotecas
O código começa com a importação das bibliotecas necessárias.

import tensorflow as tf  
from tensorflow.keras import layers, models  
from tensorflow.keras.datasets import mnist  
import matplotlib.pyplot as plt  

## Carregamento dos Dados
Os dados do MNIST são carregados e divididos em conjuntos de treinamento e teste.

(x_train, y_train), (x_test, y_test) = mnist.load_data()  


## Pré-processamento dos Dados
Os dados são normalizados para o intervalo [0, 1] e redimensionados para incluir uma nova dimensão para os canais de cor (1 canal para imagens em escala de cinza).


x_train = x_train.astype('float32') / 255  
x_test = x_test.astype('float32') / 255  
x_train = x_train.reshape((60000, 28, 28, 1))  
x_test = x_test.reshape((10000, 28, 28, 1))  

## Construção do Modelo
Um modelo sequencial de rede neural é criado com várias camadas convolucionais e de pooling, seguido de camadas densas.

model = models.Sequential()  
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  
model.add(layers.MaxPooling2D((2, 2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  
model.add(layers.MaxPooling2D((2, 2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  
model.add(layers.Flatten())  
model.add(layers.Dense(64, activation='relu'))  
model.add(layers.Dense(10, activation='softmax'))  

## Compilação do Modelo
O modelo é compilado com um otimizador, uma função de perda e métricas.

model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])  

## Treinamento do Modelo
O modelo é treinado durante um número especificado de épocas.


model.fit(x_train, y_train, epochs=5)  
Avaliação do Modelo
Após o treinamento, o modelo é avaliado no conjunto de teste.


test_loss, test_acc = model.evaluate(x_test, y_test)  
print(f'Accuracy: {test_acc}')  
Previsões
O modelo faz previsões sobre o conjunto de teste e as exibe.


predictions = model.predict(x_test)  
predicted_classes = tf.argmax(predictions, axis=1).numpy()  

## Visualização
As previsões e as imagens correspondentes são exibidas usando Matplotlib.


n_display = 10  
plt.figure(figsize=(15, 3))  
for i in range(n_display):  
    plt.subplot(2, n_display//2, i + 1)  
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')  
    plt.title(f'Pred: {predicted_classes[i]}')  
    plt.axis('off')  
plt.tight_layout()  
plt.show()  
Como Executar
Para executar este código:

## Instale as bibliotecas necessárias:

bash
pip install tensorflow matplotlib  
Copie o código para um arquivo Python e execute-o.
