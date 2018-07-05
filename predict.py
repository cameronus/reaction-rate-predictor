import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

np.set_printoptions(linewidth=100, suppress=True, precision=6)

training_x = np.loadtxt('training_x.csv', delimiter=',')
training_y = np.loadtxt('training_y.csv', delimiter=',')
testing_x = np.loadtxt('testing_x.csv', delimiter=',')
testing_y = np.loadtxt('testing_y.csv', delimiter=',')
print(testing_y)

model = Sequential()
model.add(Dropout(0.25))
model.add(BatchNormalization(input_shape=(32,)))
model.add(Dense(150, activation='linear', input_shape=(32,)))
model.add(Dense(150, activation='relu', input_shape=(150,)))
model.add(Dense(150, activation='relu', input_shape=(150,)))
model.add(Dropout(0.1))
model.add(Dense(150, activation='relu', input_shape=(150,)))
model.add(Dense(150, activation='linear', input_shape=(150,)))
model.add(Dense(1, activation='linear', input_shape=(150,)))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(training_x, training_y, epochs=1000, batch_size=256)

in_sample = model.predict(training_x).ravel()
out_of_sample = model.predict(testing_x).ravel()

print()

print(training_y)
print(in_sample)

print()

print(testing_y)
print(out_of_sample)
print(abs(testing_y - out_of_sample))
print(np.mean(abs(testing_y - out_of_sample)))
