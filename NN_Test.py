import keras
from keras.models import Sequential
from keras.layers import Dense

# Create a sequential model
model = Sequential()
model.add(Dense(2, input_shape=(2, ), activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.predict([[1, 2]]))