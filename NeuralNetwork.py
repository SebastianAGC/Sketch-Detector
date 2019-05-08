import numpy as np

# Neural Network for a 2 layer Network
class NeuralNetwork:

    def __init__(self, x, y):
        #self.input = x
        self.input = x.reshape((x.shape[0], 1))
        self.weights1 = np.random.rand(len(x), self.input.shape[0])
        self.weights2 = np.random.rand(10, len(x))
        self.y = y.reshape(y.shape[0], 1)
        self.output = np.zeros(y.shape)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return (1 / (1 + np.exp(-x)))*(1-(1 / (1 + np.exp(-x))))

    def lossfunction(self, desiredValue, obtainedValue):
        return (desiredValue-obtainedValue)**2

    def feedforward(self):
        vsigmoid = np.vectorize(self.sigmoid)
        #a1 = self.input
        #a1 = a1.reshape((a1.shape[0], 1))
        #z2 = self.weights1 @ a1
        #a2 = vsigmoid(z2)
        #self.layer1 = a2
        #z3 = self.weights2 @ self.layer1
        #a3 = vsigmoid(z3)
        #self.output = a3

        self.layer1 = vsigmoid(self.weights1 @ self.input)
        self.output = vsigmoid(self.weights2 @ self.layer1)



    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        #vsigmoid_derivative = np.vectorize(self.sigmoid_derivative)
        #d = (2 * (self.y - self.output) * vsigmoid_derivative(self.output))
        #d_weights2 = np.matmul(d, self.layer1.T)
        #a1 = self.input
        #a1 = a1.reshape((a1.shape[0], 1))
        #d_weights1 = np.matmul((np.matmul(self.weights2.T, d) * vsigmoid_derivative(self.layer1)), a1.T)

        vsigmoid_derivative = np.vectorize(self.sigmoid_derivative)
        d = (2 * (self.y - self.output) * vsigmoid_derivative(self.output))
        d_weights2 = np.matmul(d, self.layer1.T)
        d_weights1 = np.matmul((np.matmul(self.weights2.T, d) * vsigmoid_derivative(self.layer1)), self.input.T)

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
