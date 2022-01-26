import numpy as np
import pygame
from keras.datasets import mnist
from keras import models
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
import keras

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#
# network = models.Sequential()
# network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation='sigmoid'))
# network.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])

# Preprocessing training data
train_images = train_images.reshape((train_images.shape[0], 28 * 28))
train_images = train_images.astype('float32') / 255

# Preprocessing test data
test_images = test_images.reshape((test_images.shape[0], 28 * 28))
test_images = test_images.astype('float32') / 225

# network.fit(train_images, train_labels, epochs=5, batch_size=128)
network = models.load_model('model')

# Calculate Test loss and Test Accuracy
test_loss, test_acc = network.evaluate(test_images, test_labels)
network.save('./model')
# Print Test loss and Test Accuracy

# ==============================================================================================================
BIAS = 0


def activation(x):
    return x


class Adaline:
    def __init__(self, n_inputs, classified_digit, eta=0.0001, n_epoch=5):
        self.weights = np.random.random(n_inputs + 1)
        self.bias = np.random.random()
        self.inputs = []
        self.n_epoch = n_epoch
        self.eta = eta
        self.classified_digit = classified_digit

    def fit(self, features, labels):
        print(f"Perceptron nr: {self.classified_digit}")
        for epoch_index in range(self.n_epoch):
            print(f"Epoch {epoch_index}/{self.n_epoch}")
            loss = 0
            for i in range(features.shape[0]):
                random_index = np.random.randint(0, len(features))
                out = self.output(features[random_index])
                self.weights[1:] += self.eta * (labels[random_index][self.classified_digit] - out) * features[random_index]
                loss += (labels[i][self.classified_digit] - out) ** 2

            print(f"Loss: {loss/features.shape[0]}")

    def output(self, X):
        return activation(np.dot(X, self.weights[1:]) + self.weights[BIAS])

    def print_accuracy(self):
        correct = 0
        for feature, label in zip(test_images, test_labels):
            output = self.output(feature)
            if output == label[self.classified_digit]:
                correct += 1
        print(f"Test accuracy: {correct / len(test_images)}")


perceptrons = []

for i in range(10):
    perceptrons.append(Adaline(784, i, n_epoch=5))
for i, perceptron in enumerate(perceptrons):
    perceptrons[i].fit(train_images, train_labels)

n_samples = len(test_images)
correct = 0

for feature, label in zip(test_images, test_labels):
    adalines_guesses = []
    for i, perceptron in enumerate(perceptrons):
        output = perceptron.output(feature)
        adalines_guesses.append(output)
    best = adalines_guesses.index(np.max(adalines_guesses))
    if label[best] == 1:
        correct += 1

print(f"Result on test data: {correct}/{n_samples}")


# =========================== GUI ===================================



WIDTH = 28 * 15
ROWS = 28
WINDOW = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption('Digit Recognition')
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
GREEN = (40, 250, 40)


class Square:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.width = width
        self.x = row * width
        self.y = col * width
        self.total_rows = total_rows
        self.color = WHITE
        self.neighbours = []

    def get_pos(self):
        return self.row, self.col

    def isAlive(self):
        return self.color == BLACK

    def nnAlive(self):
        if self.color == BLACK:
            return 1
        else:
            return 0

    def makeAlive(self):
        self.color = BLACK

    def makeDead(self):
        self.color = WHITE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))


def makeGrid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            square = Square(i, j, gap, rows)
            grid[i].append(square)
    return grid


def drawGridlines(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
    for j in range(rows):
        pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for square in row:
            square.draw(win)

    drawGridlines(win, rows, width)
    pygame.display.update()


def get_clicked_position(mouse, rows, width):
    gap = width // rows
    y, x = mouse
    row = y // gap
    col = x // gap
    return row, col


def makeModel():
    return keras.models.load_model('model')


def nn_predict(grid, nn_running):
    nn_running = True
    inputX = []
    for row in grid:
        inputX.append([])
        for col in row:
            if col.nnAlive():
                inputX[-1].append(0.5)
            else:
                inputX[-1].append(0)

    # network.predict(np.array(inputX).reshape(784))
    for i, perceptron in enumerate(perceptrons):
        res = perceptron.output(np.array(inputX).reshape(784))
        print(f"Ama bezceptron nr {i} and i think: {res}")
    nn_running = False


def main(win, width):
    grid = makeGrid(ROWS, width)
    run = True
    nn_running = False

    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif nn_running != True and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    nn_predict(grid, nn_running)

            if not nn_running:
                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_position(pos, ROWS, width)
                    if row < len(grid) and col < len(grid):
                        square = grid[row][col]
                        square.makeAlive()
                elif pygame.mouse.get_pressed()[2]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_position(pos, ROWS, width)
                    if row < len(grid) and col < len(grid):
                        square = grid[row][col]
                        square.makeDead()

    pygame.quit()


WIDTH = 28 * 15
ROWS = 28
WINDOW = pygame.display.set_mode((WIDTH, WIDTH))

main(WINDOW, WIDTH)
