import tensorflow as tf             
from tensorflow.keras import layers, models    
from tensorflow.keras.datasets import mnist   

(x_train, y_train), (x_test, y_test) = mnist.load_data()  

x_train = x_train.astype('float32') / 255  
x_test = x_test.astype('float32') / 255  

x_train = x_train.reshape((60000, 28, 28, 1))  
x_test = x_test.reshape((10000, 28, 28, 1))  

model = models.Sequential()  
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  
model.add(layers.MaxPooling2D((2, 2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  
model.add(layers.MaxPooling2D((2, 2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  

model.add(layers.Flatten())  
model.add(layers.Dense(64, activation='relu'))  
model.add(layers.Dense(10, activation='softmax'))  

model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])  

model.fit(x_train, y_train, epochs=5)  

test_loss, test_acc = model.evaluate(x_test, y_test)  
print(f'Accuracy: {test_acc}')  

predictions = model.predict(x_test)  
predicted_classes = tf.argmax(predictions, axis=1).numpy()  

import matplotlib.pyplot as plt  
n_display = 10  
plt.figure(figsize=(15, 3))  

for i in range(n_display):  
    plt.subplot(2, n_display//2, i + 1)  
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')  
    plt.title(f'Pred: {predicted_classes[i]}')  
    plt.axis('off')  

plt.tight_layout()  
plt.show()
