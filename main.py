import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib #For saving MinMaxScaler for predictions

#load dataset from file
df = pd.read_csv(
    'SensorValues.csv',
    parse_dates=['Time'], #Namnet på den kolumnen i csv filen
    index_col='Time' #vilken kolumn som ska vara index
)

#normalize features
scaler = MinMaxScaler(feature_range=(0,1)) #Maybe change to -1 1
values = scaler.fit_transform(df.values)
#Save scaler for later
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)

#Split into train and test sets
train_size = int(len(df) * 0.8)
train = values[:train_size]
test = values[train_size:]

#Separate into input and output values (supervised learning)
train_x, train_y = train[:-1, :], train[1:, :]
test_x, test_y = test[:-1, :], test[1:, :]

#Create train and validation batches using moving window
history_size = 6*24*3 #3 days of values, size of window
x, y = [], []
for i in range(history_size, len(train)-1):
    indices = range(i-history_size, i)
    x.append(train_x[indices])
    y.append(train_y[i])
train_x, train_y = np.asarray(x), np.asarray(y)
train_batch = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_batch = train_batch.batch(16)

x, y = [], []
for i in range(history_size, len(test)-1):
    indices = range(i-history_size, i)
    x.append(test_x[indices])
    y.append(test_y[i])
test_x, test_y = np.asarray(x), np.asarray(y)
#print(train_x.shape, train_y.shape, test_x.shape, test_y.shape, sep='\n')
val_batch = tf.data.Dataset.from_tensor_slices((test_x, test_y))
val_batch = val_batch.batch(16)

#Design network
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=32),
    tf.keras.layers.Dense(units=6)
])
lstm_model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy'])

#Train network
history = lstm_model.fit(
    train_batch,
    #train_x,
    #train_y,
    epochs=10,
    batch_size=32,
    #validation_data=(test_x, test_y),
    validation_data=val_batch,
    shuffle=False
)
#Save network model
lstm_model.save('LSTM_saved_model')