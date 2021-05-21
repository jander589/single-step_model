#This script produces predictions and then evaluates them, and the model when predicting
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import joblib
import time

###########################PREPARE FOR MEASUREMENTS###########################
#Load values
df = pd.read_csv(
    'SensorValues.csv',
    parse_dates=['Time'],
    index_col='Time'
)

#Rescale values
scaler_filename="scaler.save"
scaler = joblib.load(scaler_filename)
values = scaler.transform(df.values)

#Create x and y batches
history_size = 720
test_y = values[history_size:]
predict_batch = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=values[:-1],
    targets=None,
    sequence_length=history_size,
    sequence_stride=1,
    batch_size=32,
    shuffle=False
)

#Load LSTM model
lstm_model = tf.keras.models.load_model('LSTM_saved_model')
#Predict values
yhat = lstm_model.predict(predict_batch)

#Unscale values
test_y = scaler.inverse_transform(test_y)
yhat = scaler.inverse_transform(yhat)

eval_batch = []
for i in range(history_size, len(values)-1):
    indices = range(i-history_size, i)
    eval_batch.append(values[indices])
eval_batch = np.array(eval_batch)

###########################MEASURE PERFORMANCE###########################
#Good site for evaluation --> https://otexts.com/fpp2/accuracy.html

#Calculate error
#Temp error
temp_error = []
for i in range(test_y.shape[0]):
    temp_error.append(abs(test_y[i, 0]-yhat[i, 0]))

#Humidity error
hum_error = []
for i in range(test_y.shape[0]):
    hum_error.append(abs(test_y[i, 1]-yhat[i, 1]))

#Pressure error
press_error = []
for i in range(test_y.shape[0]):
    press_error.append(abs(test_y[i, 2]-yhat[i, 2]))

#Gas error
gas_error = []
for i in range(test_y.shape[0]):
    gas_error.append(abs(test_y[i, 3]-yhat[i, 3]))

#Lux error
lux_error = []
for i in range(test_y.shape[0]):
    lux_error.append(abs(test_y[i, 4]-yhat[i, 4]))

#co2 error
co_error = []
for i in range(test_y.shape[0]):
    co_error.append(abs(test_y[i, 5]-yhat[i, 5]))

#Calculate mean of all errors
temp_mean = np.mean(temp_error)
hum_mean = np.mean(hum_error)
press_mean = np.mean(press_error)
gas_mean = np.mean(gas_error)
lux_mean = np.mean(lux_error)
co_mean = np.mean(co_error)

#Calculate standard deviation of all errors
temp_std = np.std(temp_error)
hum_std = np.std(hum_error)
press_std = np.std(press_error)
gas_std = np.std(gas_error)
lux_std = np.std(lux_error)
co_std = np.std(co_error)

#Calculate average time and std to predict a value
times = []
n = 100
for i in range(n):
    predic = values[i:i+history_size]
    predic = np.asarray(predic)
    predic = predic.reshape(1, predic.shape[0], predic.shape[1])
    t0 = time.time()
    lstm_model.predict(predic)
    t1 = time.time()
    times.append(t1-t0)
time_mean = np.mean(times)
time_std = np.std(times)

#Print all values
print(f'Time mean={time_mean} std={time_std}')
print(f'Temp mean={temp_mean} std={temp_std}')
print(f'Hum mean={hum_mean} std={hum_std}')
print(f'Press mean={press_mean} std={press_std}')
print(f'Gas mean={gas_mean} std={gas_std}')
print(f'Lux mean={lux_mean} std={lux_std}')
print(f'co2 mean={co_mean} std={co_std}')