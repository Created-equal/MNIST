import numpy as np
import random

def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
        return sigmoid(x) * (1.0 - sigmoid(x))

class Neural_Network():
        def __init__(self, layer_size):
                self.size = layer_size
                self.bias = [np.random.randn(s, 1)
                                for s in self.size[1 :]]
                self.weight = [np.random.randn(s2, s1)
                                for (s1, s2) in zip(self.size, self.size[1 :])]
        
        def feedforward(self, x):
                for (w, b) in zip(self.weight, self.bias):
                        x = sigmoid(np.dot(w, x) + b)
                return x

        def evaluate(self, test_data):
                Cnt = 0
                for test in test_data:
                        if np.argmax(self.feedforward(test[0])) == test[1]:
                                Cnt = Cnt + 1
                print('accuracy : %d / %d' % (Cnt, len(test_data)))
        
        def train(self, training_data, minibatch_size, eta):
                (n, Data) = (len(training_data), list(training_data))
                random.shuffle(Data)
                for k in range(0, n, minibatch_size):
                        self.update(Data[k : k + minibatch_size], eta)

        def backpropagation(self, x, y):
                output = [x]
                before = []
                for (w, b) in zip(self.weight, self.bias):
                        before.append(np.dot(w, output[-1]) + b)
                        output.append(sigmoid(before[-1]))
                gradient_bias = [np.zeros(np.shape(b))
                                for b in self.bias]
                gradient_weight = [np.zeros(np.shape(w))
                                for w in self.weight]
                L = len(gradient_bias)
                gradient_bias[-1] = (output[-1] - y) * d_sigmoid(before[-1])
                gradient_weight[-1] = np.dot(gradient_bias[-1], output[-2].transpose())
                for l in range(2, L + 1):
                        gradient_bias[-l] = np.dot(self.weight[-(l - 1)].transpose(), gradient_bias[-(l - 1)]) * d_sigmoid(before[-l])
                        gradient_weight[-l] = np.dot(gradient_bias[-l], output[-(l + 1)].transpose())
                return (gradient_bias, gradient_weight)
        
        def update(self, Data, eta):
                gradient_bias = [np.zeros(np.shape(b))
                                for b in self.bias]
                gradient_weight = [np.zeros(np.shape(w))
                                for w in self.weight]
                for case in Data:
                        (temp_bias, temp_weight) = self.backpropagation(case[0], case[1])
                        L = len(self.size) - 1
                        for i in range(L):
                                gradient_bias[i] += temp_bias[i]
                        for i in range(L):
                                gradient_weight[i] += temp_weight[i]
                N = len(Data)
                L = len(self.size) - 1
                for i in range(L):
                        gradient_bias[i] /= N
                        self.bias[i] -= gradient_bias[i] * eta
                for i in range(L):
                        gradient_weight[i] /= N
                        self.weight[i] -= gradient_weight[i] * eta

def main():
        import mnist_loader
        (training_data, validation_data, test_data) = mnist_loader.load_data_wrapper()
        training_data = tuple(training_data)
        validation_data = tuple(validation_data)
        test_data = tuple(validation_data)
        network = Neural_Network((784, 50, 10))
        """
                Hyper-parameters of each training-epoch are decided by users.
                minibatch_size : the size of each minibatch in this training-epoch
                eta : the learning rate of this training-epoch
        """
        while True:
                network.train(training_data, int(input("minibatch_size : ")), float(input("eta : ")))
                network.evaluate(test_data)

main()