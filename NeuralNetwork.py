import numpy as np

# Neural Network for a 2 layer Network
class NeuralNetwork:

    def __init__(self):
        self.weights1 = np.random.randn(100, 784)
        self.weights2 = np.random.randn(10, 100)
        self.bias1 = np.random.randn(100, 1)
        self.bias2 = np.random.randn(10, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, x):
        z1 = (self.weights1 @ x) + self.bias1
        layer1 = self.sigmoid(z1)
        z2 = (self.weights2 @ layer1) + self.bias2
        return self.sigmoid(z2)

    def success(self, cross):
        i = 0
        for x, y in cross:
            index = np.argmax(self.feedforward(x))
            if (y[index] == 1):
                i += 1
        return i

    def gradientDescent(self, dataset, cross, test, rate, iters):
        k = rate / len(dataset)
        for i in range(iters):
            print(i)
            gw1 = np.zeros(self.weights1.shape)
            gw2 = np.zeros(self.weights2.shape)
            gb1 = np.zeros(self.bias1.shape)
            gb2 = np.zeros(self.bias2.shape)
            for x, y in dataset:
                grad_w1, grad_w2, grad_b1, grad_b2 = self.backprop(x, y)
                gw1 += grad_w1
                gw2 += grad_w2
                gb1 += grad_b1
                gb2 += grad_b2
            self.weights1 -= k * gw1
            self.weights2 -= k * gw2
            self.bias1 -= k * gb1
            self.bias2 -= k * gb2
            print("Cross: {}%".format(self.success(cross) / len(cross) * 100))
            print("Test: {}%".format(self.success(test) / len(test) * 100))

    def backprop(self, x, y):
        a1 = self.sigmoid(self.weights1 @ x + self.bias1)
        a2 = self.sigmoid(self.weights2 @ a1 + self.bias2)
        d2 = a2 - y
        d1 = self.weights2.T @ d2 * a1 * (1 - a1)
        return (d1 @ x.T, d2 @ a1.T, d1, d2)