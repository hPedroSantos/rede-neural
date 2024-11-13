import tensorflow as tf             # Importa a biblioteca TensorFlow, que é usada para criar e treinar redes neurais  
from tensorflow.keras import layers, models   # Importa módulos para criar camadas de rede neural e construir modelos  
from tensorflow.keras.datasets import mnist   # Importa o conjunto de dados MNIST  

# Carregar os dados  
(x_train, y_train), (x_test, y_test) = mnist.load_data()  
# Carrega os dados MNIST e os divide em conjuntos de treinamento (x_train, y_train) e teste (x_test, y_test)  

# Normalizar os dados  
x_train = x_train.astype('float32') / 255    # Converte os dados de treinamento para float e normaliza para o intervalo [0, 1]  
x_test = x_test.astype('float32') / 255      # Faz o mesmo para os dados de teste  

# Adicionar uma dimensão para a imagem  
x_train = x_train.reshape((60000, 28, 28, 1))  # Redimensiona para incluir uma dimensão para canais (1 canal para escala de cinza)  
x_test = x_test.reshape((10000, 28, 28, 1))    # Faz o mesmo para os dados de teste  

# Criar o modelo da rede neural  
model = models.Sequential()                       # Inicializa um modelo sequencial, que é uma pilha de camadas  
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  
# Adiciona uma camada convolucional com 32 filtros, tamanho de kernel (3, 3), função de ativação 'relu' e formato de entrada de (28, 28, 1)  

model.add(layers.MaxPooling2D((2, 2)))                       # Adiciona uma camada de pooling para reduzir a dimensionalidade  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))      # Adiciona mais uma camada convolucional  
model.add(layers.MaxPooling2D((2, 2)))                       # Adiciona outra camada de pooling  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))      # Adiciona mais uma camada convolucional  

model.add(layers.Flatten())                                   # Achata a saída da camada anterior para uma dimensão  
model.add(layers.Dense(64, activation='relu'))                # Adiciona uma camada densa com 64 neurônios e função de ativação 'relu'  
model.add(layers.Dense(10, activation='softmax'))             # Adiciona a camada de saída com 10 neurônios (um para cada dígito) e ativação 'softmax'  

# Compilar o modelo  
model.compile(optimizer='adam',                             # Compila o modelo usando o otimizador 'adam'  
              loss='sparse_categorical_crossentropy',      # Define a função de perda como entropia cruzada esparsa para múltiplas classes  
              metrics=['accuracy'])                          # Especifica que queremos acompanhar a acurácia durante o treinamento  

# Treinar o modelo  
model.fit(x_train, y_train, epochs=5)                       # Treina o modelo em 5 épocas (ciclos completos através dos dados de treinamento)  

# Avaliar o modelo  
test_loss, test_acc = model.evaluate(x_test, y_test)       # Avalia o modelo nos dados de teste e obtém perda e acurácia  
print(f'Accuracy: {test_acc}')                             # Imprime a acurácia no conjunto de teste  

# Fazer previsões  
predictions = model.predict(x_test)                          # Faz previsões para o conjunto de teste  
predicted_classes = tf.argmax(predictions, axis=1).numpy()  # Converte as previsões em classes (números de 0 a 9)  

# Exibir algumas previsões  
import matplotlib.pyplot as plt                               # Importa a biblioteca para visualização  
n_display = 10  # Número de imagens a exibir  
plt.figure(figsize=(15, 3))  

for i in range(n_display):  
    plt.subplot(2, n_display//2, i + 1)                      # Cria uma grade de subgráficos  
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')      # Exibe a imagem original em escala de cinza  
    plt.title(f'Pred: {predicted_classes[i]}')              # Adiciona o título com a previsão do modelo  
    plt.axis('off')                                          # Remove os eixos  

plt.tight_layout()                                           # Ajusta a disposição dos subgráficos  
plt.show()                                                  # Exibe a figura
