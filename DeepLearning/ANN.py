import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ReLU, LeakyReLU, PReLU, ELU
from keras.layers import Dropout
from keras.optimizers import Adam

print(tf.__version__)

dataset = pd.read_csv('../Datasets/Churn_Modelling.csv')
print(dataset.head())

X = dataset.iloc[:, 3: 13]
y = dataset.iloc[:, 13]

# Feature Engineering - Create dummy variables
geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

# Concatenate the Data Frames
X = pd.concat([X, geography, gender], axis=1)
print(X.head())

# Drop Unnecessary columns
X = X.drop(['Geography', 'Gender'], axis=1)
print(X.head())

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape)

# Part 2 - Now let's make the ANN!
# Initialising the ANN
classifier = Sequential()

# Adding the input layer (11 columns in 'X' means 11 Inputs, so units=11)
classifier.add(Dense(units=11, activation='relu'))
classifier.add(Dropout(0.2))

# Adding the first hidden layer
classifier.add(Dense(units=7, activation='relu'))

# Adding the second hidden layer
classifier.add(Dense(units=6, activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, activation='sigmoid'))

opt = Adam(learning_rate=0.01)

classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=50,
                               callbacks=early_stopping)
print(model_history.history.keys())

# Evaluate The Model
train_loss, train_acc = classifier.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = classifier.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
y_pred = y_pred > 0.5

# Making the Confusion Matrix
c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix)

# Calculate the Accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Get the weights
weights = classifier.get_weights()
# print(weights)
