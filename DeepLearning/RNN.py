import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

dataset_train = pd.read_csv('../Datasets/Google_Stock_Price_Train.csv')
print(dataset_train.head())

train = dataset_train.loc[:, ['Open']].values
print(train)

# Feature Scaling
scaler = MinMaxScaler()
trained_scaled = scaler.fit_transform(train)
print(trained_scaled)

# plt.plot(trained_scale)
# plt.show()

# Create Data Structure
X_train = []
y_train = []
timesteps = 50

for i in range(timesteps, 1250):
    X_train.append(trained_scaled[i - timesteps:i, 0])
    y_train.append(trained_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Reshaping

# Create RNN Model
# Initialize RNN
regressor = Sequential()

# Adding the first RNN layer and some Dropout regularization
regressor.add(SimpleRNN(units=50, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding the second RNN layer and some Dropout regularization
regressor.add(SimpleRNN(units=50, activation='tanh', return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the third RNN layer and some Dropout regularization
regressor.add(SimpleRNN(units=50, activation='tanh', return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the fourth RNN layer and some Dropout regularization
regressor.add(SimpleRNN(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compile the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
history = regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Prediction and Visualization of RNN Model
dataset_test = pd.read_csv('../Datasets/Google_Stock_Price_Test.csv')
print(dataset_test.head())

real_stock_price = dataset_test.loc[:, ['Open']].values
print(real_stock_price)

# Getting the predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1, 1)
inputs = scaler.transform(inputs)
print(inputs)

X_test = []
for i in range(timesteps, 70):
    X_test.append(inputs[i - timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshaping

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
