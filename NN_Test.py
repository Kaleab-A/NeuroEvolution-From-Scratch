# import keras
# from keras.models import Sequential
# from keras.layers import Dense

# # Create a sequential model
# model = Sequential()
# model.add(Dense(2, input_shape=(2, ), activation='relu'))
# model.add(Dense(2, activation='relu'))
# model.add(Dense(1, activation='relu'))
# model.add(Dense(2, activation='relu'))
# model.add(Dense(2, activation='sigmoid'))

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print(model.predict([[1, 2]]))

# Read 2 files NN1.txt and NN2.txt
# Each have list of number on new line 
# Plot a line graph with 2 lines

from matplotlib import pyplot as plt

def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        lines = [float(line.strip()) for line in lines]
        return lines

def plot_graph(x, y1, y2):
    plt.plot(x, y1, label='NN1')
    # plt.plot(x, y2, label='NN2')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.show()

def main():
    x = [i for i in range(1, 101)]
    y1 = read_file('NN1.txt')
    y2 = read_file('NN2.txt')
    plot_graph(x, y1, y2)

if __name__ == '__main__':
    main()