import numpy as np

# Original info: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# 2-layer Neural Network (1 hidden and 1 output)


# The Sigmoid function, which describes an S shaped curve.
# We pass the weighted sum of the inputs through this function to
# normalise them between 0 and 1.
def sigmoid(x):
    # Sigmoid: f(x) = 1 / 1 + e(-x)
    return 1.0 / (1 + np.exp(-x))


# define the derivative of our Sigmoid function to use later for back-prop
# The derivative of the Sigmoid function.
# This is the gradient of the Sigmoid curve.
# It indicates how confident we are about the existing weight.
def sigmoid_derivative(x):
    return x * (1.0 - x)


# Neural Networks consist of the following components
# An input layer, x
# An arbitrary amount of hidden layers
# An output layer, ŷ
# A set of weights and biases between each layer, W and b
# A choice of activation function for each hidden layer, σ. In this tutorial, we’ll use a Sigmoid activation function.


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        # weight - связи
        # We model a single neuron, with 4 input connections and 1 output connection.
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        # initializing our Synapses
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    # Use feedforward neural networks if you want to deal with non-linearly separable data:
    # https://hackernoon.com/building-a-feedforward-neural-network-from-scratch-in-python-d3526457156b
    # Calculating the predicted output ŷ
    def feedforward(self):
        # Нейронные сети прямого распространения (feed forward neural networks, FF или FFNN)
        # передают информацию от входа к выходу
        # Этот процесс называется обучением с учителем, и он отличается от обучения без учителя тем, что во втором случае
        # множество выходных данных сеть составляет самостоятельно.
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    # Updating the weights and biases
    def backprop(self):
        # Алгоритм Back Propagation для понимания, насколько и в какую сторону смещать веса
        # считает ошибку с конца — справа (от результата) налево (до первого слоя нейронов).

        # Calculate the error (The difference between the desired output and the predicted output).
        error = self.y - self.output
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (error) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (error) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    # using 4 vectors (w0, w1, w2, w3) for our training set. Input dataset: X0, X1, X2
    # w0
    # w1
    # w2
    # w3
    X = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    # output dataset
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # Input [0, 0, 1] => Output [0]
    # Input [0, 1, 1] => Output [1]
    # Input [1, 0, 1] => Output [1]
    # Input [1, 1, 1] => Output [1]

    nn = NeuralNetwork(X, y)

    for iteration in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)