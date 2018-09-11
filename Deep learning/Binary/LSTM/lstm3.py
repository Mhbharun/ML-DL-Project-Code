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

numpy.random.seed(3)

# To read the train data and normalization
traindata = pd.read_csv('data/train10.csv', header=None)
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(traindata)
train = np.reshape(train, (train.shape[0], 1, train.shape[1]))

# To read trainlabel and normalization
trainlabel = pandas.read_csv('data/trainlabel10.csv', header=None)
scaler = MinMaxScaler(feature_range=(0, 1))
train_label = scaler.fit_transform(trainlabel)

# Both train and train_label are converted in to arrays
train = np.array(train)

train_label = np.array(train_label)


model = Sequential()
model.add(LSTM(32, input_dim=1, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer='rmsprop')

# Train
checkpointer = callbacks.ModelCheckpoint(filepath="logs/lstm3layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('logs/lstm3layer/lstmtrainanalysis1.csv',separator=',', append=False)
model.fit(train, train_label, nb_epoch=1000, batch_size=32, callbacks=[checkpointer,csv_logger])
model.save("logs/lstm31layer/lstm_model.hdf5")

