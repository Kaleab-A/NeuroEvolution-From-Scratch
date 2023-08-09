import keras
from keras.models import Sequential
from keras.layers import Dense
import math

# Create a sequential model
model = Sequential()
model.add(Dense(2, input_shape=(1, ), activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mse')

x = []
y = []

def f(x):
    return math.sin(math.radians(x))

a = 0
while a < 90:
    x.append(a)
    y.append(f(a))
    a += 0.01

out = model.predict([0, 30, 45, 60, 90])
print(out)


model.fit(x, y, epochs=10, batch_size=16, shuffle=True)

out2 = model.predict([0, 30, 45, 60, 90])
print(out2)





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