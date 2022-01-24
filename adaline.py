import numpy as np
import random

def fourier_transform(x):
  a = np.abs(np.fft.fft(x))
  a[0] = 0
  return a/np.amax(a)


class Adaline(object):

  def __init__(self, no_of_inputs, learning_rate=0.01, iterations=2000, sigma = True):
    self.no_of_inputs = no_of_inputs
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.weights = np.random.random(2*self.no_of_inputs) # przypisujemy wagom male losowe wartosci
    self.sigma = sigma
    self.errors = []

  def train(self, przyklad, odpowiedz):
    for index in range(self.iterations):
        e = 0
        print(f"Epoch {index}/{self.iterations}")

        for x, y in random.choices(list(zip(przyklad, odpowiedz))):
            # losowo dobieramy przyklad uczący i odp
            x = np.concatenate([x, fourier_transform(x)])

            # obliczamy aktywacje jednostki na przedziale e=+1
            out = self.output(x)

            #korygujemy wagi
            self.weights += self.learning_rate * (y - out) * x

            e += (y - out) ** 2

        self.errors.append(e)
        print(f"Loss: {e}")

  def _activation(self, x):
      x = 0.8 * x + 0.1
      return 1 / (1 + np.exp(-x))


  def output(self, x):
    return self._activation(np.dot(self.weights, x))
