import tensorflow as tf
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import joblib #For loading MinMaxScaler for predictions

#Load all values
df = pd.read_csv(
    'SensorValues.csv',
    parse_dates=['Time'],
    index_col='Time'
)
values = df.values[8142:, :] #Save a month worth of values

#Rescale the values
scaler_filename = "scaler.save"
scaler = joblib.load(scaler_filename)
values = scaler.transform(values)

#Save for comparing
test_y = values[1:, :]
history_size = 720
predict_batch = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=values[:-1,:],
    targets=None,
    sequence_length=history_size,
    sequence_stride=1,
    batch_size=32,
    shuffle=False
)

#Load the saved lstm model and predict valeus
lstm_model = tf.keras.models.load_model('LSTM_saved_model')
yhat = lstm_model.predict(predict_batch)

test_y = scaler.inverse_transform(test_y)
yhat = scaler.inverse_transform(yhat)

test_y = test_y[history_size:]
#Print values
font = {'family': 'normal', 'weight' : 'bold', 'size' : 15}
pyplot.rc('font', **font)
fig, axs = pyplot.subplots(3, 2)
axs[0,0].plot(test_y[:, 0], color='b', label='Actual value')
axs[0,0].plot(yhat[:, 0], color='r', label='Predicted value')
axs[0,0].set_title('Temperature')
axs[0,0].set_ylim([19,23])
axs[0,1].plot(test_y[:, 1], color='b', label='Actual value')
axs[0,1].plot(yhat[:, 1], color='r', label='Predicted value')
axs[0,1].set_title('Humidity')
axs[0,1].set_ylim([18,50])
axs[1,0].plot(test_y[:, 2], color='b', label='Actual value')
axs[1,0].plot(yhat[:, 2], color='r', label='Predicted value')
axs[1,0].set_title('Pressure')
axs[1,0].set_ylim([942,1000])
axs[1,1].plot(test_y[:, 3], color='b', label='Actual value')
axs[1,1].plot(yhat[:, 3], color='r', label='Predicted value')
axs[1,1].set_title('gas/1000')
axs[1,1].set_ylim([8,1700])
axs[2,0].plot(test_y[:, 4], color='b', label='Actual value')
axs[2,0].plot(yhat[:, 4], color='r', label='Predicted value')
axs[2,0].set_title('Lux')
axs[2,0].set_ylim([-10,75])
axs[2,1].plot(test_y[:, 5], color='b', label='Actual value')
axs[2,1].plot(yhat[:, 5], color='r', label='Predicted value')
axs[2,1].set_title('co2')
axs[2,1].set_ylim([300,1200])
pyplot.subplots_adjust(hspace=0.5)
pyplot.legend(loc='upper center', bbox_to_anchor=(-0.1, 4.5))
pyplot.show()