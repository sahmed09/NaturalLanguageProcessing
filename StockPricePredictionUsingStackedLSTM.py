import math
import pandas_datareader as pdr  # pip install pandas-datareader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

print(tf.__version__)

"""Data Collection"""
# key = ""
# df = pdr.get_data_tiingo('AAPL', api_key=key)
# df.to_csv('Datasets/AAPL.csv')

df = pd.read_csv('Datasets/AAPL.csv')
print(df.head())
print(df.tail())

df1 = df.reset_index()['close']
print(df1)
print(df1.shape)

# plt.plot(df1)
# plt.show()

# LSTM are sensitive to the scale of the data. so we apply MinMax scaler
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
print(df1.shape)
print(df1)

# splitting dataset into train and test split
# For Time series data -> we should split the data based on date. Avoid using cross validation or random seed
# As next data is dependent on previous data
train_size = int(len(df1) * 0.65)
test_size = len(df1) - train_size
train_data, test_data = df1[0:train_size, :], df1[train_size:len(df1), :1]
print(train_size, test_size)
print(len(train_data), len(test_data))
# print(train_data)
# print(test_data)


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  # i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# reshape input (3 dimension) to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(X_train.shape, X_test.shape)

# Create the Stacked LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Lets Do the prediction and check performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE performance metrics
print(math.sqrt(mean_squared_error(y_train, train_predict)))

# Test Data RMSE
print(math.sqrt(mean_squared_error(y_test, test_predict)))

# Plotting
# shift train predictions for plotting
look_back = 100
train_predict_plot = np.empty_like(df1)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict
# shift test predictions for plotting
test_predict_plot = np.empty_like(df1)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()

print(len(test_data))

x_input = test_data[341:].reshape(1, -1)
print(x_input.shape)

temp_input = list(x_input)
temp_input = temp_input[0].tolist()
print(temp_input)

# demonstrate prediction for next 30 days
lst_output = []
n_steps = 100
i = 0
while i < 30:
    if len(temp_input) > 100:
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        print('{} day input {}'.format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape(1, n_steps, 1)
        # print(x_input)
        y_hat = model.predict(x_input, verbose=0)
        print('{} day output {}'.format(i, y_hat))
        temp_input.extend(y_hat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(y_hat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape(1, n_steps, 1)
        y_hat = model.predict(x_input, verbose=0)
        print(y_hat[0])
        temp_input.extend(y_hat[0].tolist())
        print(len(temp_input))
        lst_output.extend(y_hat.tolist())
        i = i + 1

print(lst_output)

day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)

print(len(df1))

plt.plot(day_new, scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))
plt.show()

df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1000:])
plt.show()

df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])
plt.show()

df3 = scaler.inverse_transform(df3).tolist()
plt.plot(df3)
plt.show()
