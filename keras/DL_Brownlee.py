'''
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
'''
# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
split = int(dataset.shape[0] * 0.8)
X = dataset[0:split,0:8]
X_test = dataset[split:,0:8]
y = dataset[:split,8]
y_test = dataset[split:,8]
# define the keras model
model = Sequential()
model.add(Dense(24, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs = 150, batch_size=10, verbose = 1)

_, accuracy1 = model.evaluate(X,y)
_, accuracy2 = model.evaluate(X_test, y_test)

print(accuracy1, accuracy2)

















# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # compile the keras model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # fit the keras model on the dataset
# model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# # make class predictions with the model
# _, accuracy = model.evaluate(X_test, y_test)
# print('Accuracy: %.2f' % (accuracy*100))
#
# #### another method of finding accuracy ########
# # from sklearn.metrics import accuracy_score
# # y_pred = model.predict_classes(X_test)
# # score = accuracy_score(y_test, y_pred)
# # print(score)
