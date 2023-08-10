import keras
from keras.models import Sequential
from keras.layers import Dense
import math
import numpy as np
from keras.optimizers import SGD, Adam

# Create a sequential model
model = Sequential()
model.add(Dense(8, input_shape=(1,), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="linear"))


# Compile the model
model.compile(
    # optimizer=SGD(
    #     lr=0.01,
    #     decay=0,
    # ),
    optimizer=Adam(),
    loss="mse",
)

# x = [[3, 4], [7, 14], [8, 1], [3, 3], [20, 12], [5, 6]]

# y = [[7], [21], [9], [6], [32], [11]]

x = []
y = []


def f(x):
    return x**2 + 3 * x + 1


a = 0
while a < 30:
    x.append(a)
    y.append(f(a))
    a += 0.01

# out = model.predict([0, 30, 45, 60, 90])
# print(out)

for i in range(10):
    print("Epoch: ", i)
    history = model.fit(x, y, epochs=1, batch_size=16, verbose=1, shuffle=True)

# ----------------------------------------------------------------------------------------------------


# Read 2 files NN1.txt and NN2.txt
# Each have list of number on new line
# Plot a line graph with 2 lines

# from matplotlib import pyplot as plt

# def read_file(file_name):
#     with open(file_name, 'r') as f:
#         lines = f.readlines()
#         lines = [float(line.strip()) for line in lines]
#         return lines

# def plot_graph(x, y1, y2):
#     plt.plot(x, y1, label='NN1')
#     # plt.plot(x, y2, label='NN2')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy vs Epochs')
#     plt.legend()
#     plt.show()

# def main():
#     x = [i for i in range(1, 101)]
#     y1 = read_file('NN1.txt')
#     y2 = read_file('NN2.txt')
#     plot_graph(x, y1, y2)

# if __name__ == '__main__':
#     main()
