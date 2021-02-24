'''
https://medium.com/analytics-vidhya/deep-learning-tutorial-with-keras-7a34a1a322cd
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
model = Sequential()
# print(model)

# 1. Import the imdb dataset
import keras
from keras.datasets import imdb
imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(x_train.shape)
print(x_test.shape)

from keras.preprocessing.sequence import pad_sequences
#pad all input sequences to have the length of 100
X_train = pad_sequences(x_train, maxlen=100)
X_test = pad_sequences(x_test, maxlen=100)


# 2. Create the model
#import Sequential model
from keras import models
#create Keras model
model = models.Sequential()

# 3. adding layers
# a convolutional layer is best suited for data that contain images
# an Embedding layer is best suited for data that contains text.
from keras.layers import Embedding
# first hidden layer is an embedding layer that converts each word into word vector
model.add(Embedding(input_dim=10000, output_dim=64))

# Next, add the LSTM layer found in keras.layers.LSTM. LSTM layers are preferred because they can learn long-term
# dependencies of data and thus make accurate predictions.
# import LSTM layer
from keras.layers import LSTM
# second hidden layer is an LSTM layer
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))

# Lastly, you will add a simple dense layer from keras.layers.Dense. A dense layer is a type of layer that connects
# each input to each output within its layer.
from keras.layers import Dense
model.add(keras.layers.Dense(1, activation='sigmoid'))

# 4. Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])


# 5. Training
# fit model on training set and check the accuracy on validation set.
model.fit(X_train, y_train, batch_size=20, epochs=1,
          validation_data=(X_test, y_test), verbose = 1)

_, accuracy1 = model.evaluate(X_train,y_train, verbose=1)
_, accuracy2 = model.evaluate(X_test, y_test, verbose=1)

print(accuracy1, accuracy2)


