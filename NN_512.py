import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers


def build_model():
    model = Sequential()
    model.add(Dense(256, input_dim=512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def file_read(name):
    name = str(name)
    file = pd.read_csv(name, header=None, delimiter=' ')
    file = file.values
    return file


data_low = file_read('n=16_sin_cos/T=0.01.txt')
data_high = file_read('n=16_sin_cos/T=2.0.txt')


labels_low = np.zeros(data_low.shape[0]).reshape((data_low.shape[0], 1))
labels_high = np.ones(data_high.shape[0]).reshape((data_high.shape[0], 1))

X = np.concatenate((data_low, data_high), axis=0)
y = np.concatenate((labels_low, labels_high), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

model = build_model()
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=2)

scores = model.evaluate(X_test, y_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

a = np.round(model.predict(X_test))
y_test = y_test.reshape(20000, 1)
np.savetxt("a.txt", a)
np.savetxt("y_test.txt", y_test)

test = np.concatenate((a, y_test), axis=1)
scores2 = np.sum(test[:, 0] == test[:, 1])
print(scores2)


model_json = model.to_json()
with open("model.json", 'w') as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
