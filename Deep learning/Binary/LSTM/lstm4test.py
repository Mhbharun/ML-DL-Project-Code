import numpy
import pandas
from sklearn import preprocessing
from sklearn import cross_validation
from matplotlib import pyplot
import pandas as pd
import numpy as np
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras import callbacks
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.layers.recurrent import LSTM
import math
from sklearn.metrics import mean_squared_error
numpy.random.seed(3)

# To read the train data and normalization
traindata = pd.read_csv('data/test10.csv', header=None)
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(traindata)
train = np.reshape(train, (train.shape[0], 1, train.shape[1]))

# To read trainlabel and normalization
trainlabel = pandas.read_csv('data/testlabel10.csv', header=None)
scaler = MinMaxScaler(feature_range=(0, 1))
train_label = scaler.fit_transform(trainlabel)

# Both train and train_label are converted in to arrays
test = np.array(train)

dataoriginal= np.array(train_label)


model = Sequential()
model.add(LSTM(32, input_dim=1, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer='rmsprop')

# Train
score = []
name = []
import os
for file in os.listdir("logs/lstm4layer/"):
  model.load_weights("logs/lstm4layer/"+file)
  testPredict = model.predict(test)
  # invert predictions
  inv_testPredict = scaler.inverse_transform(testPredict)
  trainScore = math.sqrt(mean_squared_error(dataoriginal, inv_testPredict))
  #print('Test Score: %.2f RMSE' % (trainScore))
  score.append(trainScore)
  name.append(file)

print(min(score))
print(name[score.index(min(score))])


model.load_weights("logs/lstm4layer/" +name[score.index(min(score))])
testPredict = model.predict(test)


# invert predictions
inv_testPredict = scaler.inverse_transform(testPredict)
trainScore = math.sqrt(mean_squared_error(dataoriginal, inv_testPredict))
print('Test Score: %.2f RMSE' % (trainScore))


np.savetxt("res/lstm4layer/lstm4layer_exact.txt", dataoriginal)
np.savetxt("res/lstm4layer/lstm4layer_predict.txt", inv_testPredict)


