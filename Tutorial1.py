import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#create model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
    ])

#set peramiters
model.compile(Optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#train machine
model.fit(train_images, train_labels, epochs=5)

#test models accuracy and loss
#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("Tested Acc:", test_acc)

prediction= model.predict(test_images[7])

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap='gray')
    plt.xlabel("Actual: " + class_names[test_labels[i]])#error list index out of range
    plt.title("Prediction" + class_names[np.argmax(prediction[i])])
    plt.show()
