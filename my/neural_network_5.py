import numpy as np
import random

def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
        return sigmoid(x) * (1.0 - sigmoid(x))

class Quadratic():
        def delta(self, output, before, qwq):
                return (output - qwq) * d_sigmoid(before)

#Cross-entropy cost function is assigned to address the learning slowdown
class CrossEntropy():
        def delta(self, output, before, qwq):
                return (output - qwq)

class Neural_Network():
        def __init__(self, layer_size, cost):
                self.size = layer_size
                """
                By Michael Nielsen:
                Initialize each weight using a Gaussian distribution with mean 0 and standard deviation 1 over the square root of the number of weights connecting to the same neuron.
                Initialize the biases using a Gaussian distribution with mean 0 and standard deviation 1.
                """
                self.bias = [np.random.randn(s, 1)
                                for s in self.size[1 :]]
                #change the way of weight-initialization to avoid a learning slowdown
                self.weight = [np.random.randn(s2, s1) / np.sqrt(s1)
                                for (s1, s2) in zip(self.size, self.size[1 :])]
                #the cost function is decided by users
                self.cost = cost
        
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
                return Cnt
        
        def train(self, training_data, minibatch_size, eta, lmbda = 0.0):
                #lmbda : The parameter lambda is used for L2-regularization, which is to overcome overfitting
                (N, Data) = (len(training_data), list(training_data))
                random.shuffle(Data)
                for i in range(0, N, minibatch_size):
                        self.update(Data[i : i + minibatch_size], eta, lmbda / N)

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
                gradient_bias[-1] = self.cost.delta(output[-1], before[-1], y)
                gradient_weight[-1] = np.dot(gradient_bias[-1], output[-2].transpose())
                for l in range(2, L + 1):
                        gradient_bias[-l] = np.dot(self.weight[-(l - 1)].transpose(), gradient_bias[-(l - 1)]) * d_sigmoid(before[-l])
                        gradient_weight[-l] = np.dot(gradient_bias[-l], output[-(l + 1)].transpose())
                return (gradient_bias, gradient_weight)
        
        def update(self, Data, eta, lmbda = 0.0):
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
                        gradient_weight[i] = gradient_weight[i] / N + lmbda * gradient_weight[i]
                        self.weight[i] -= gradient_weight[i] * eta

def main():
        import mnist_loader
        (training_data, validation_data, test_data) = mnist_loader.load_data_wrapper()
        training_data = tuple(training_data)
        validation_data = tuple(validation_data)
        test_data = tuple(validation_data)
        #The number of hidden neurons is decided by users
        hidden_neurons = 100#int(input('The number of hidden neurons : '))
        cost = Quadratic()
        while True:
                """
                1 : Quadratic Cost Function
                2 : Cross-entropy Cost Function
                Otherwise : choose again
                """
                which = 2#int(input("Cost function : "))
                if which == 1 or which == 2 :
                        if which == 2 :
                                cost = CrossEntropy()
                        break
        lmbda = 5.0#float(input('The regularization parameter lambda used for L2-regularization : '))
        network = Neural_Network((784, hidden_neurons, 10), cost)
        """
                minibatch_size : the size of each minibatch in this training-epoch
                eta : the learning rate of this training-epoch
        """
        eta = 1.0
        history_accuracy = 0
        lose_cnt = 0
        #a variable learning schedule : slow down the training(namely, reduce the learning rate) gradully
        while True:
                #network.train(training_data, int(input('minibatch_size : ')), float(input('learning rate : ')), lmbda)
                print('now eta : %.8lf' % eta)
                network.train(training_data, 10, eta, lmbda)
                now_accuracy = network.evaluate(test_data)
                if now_accuracy <= history_accuracy :
                        lost_cnt = lost_cnt + 1
                else :
                        lost_cnt = 0
                        history_accuracy = now_accuracy
                if lost_cnt == 3 :
                        eta /= 2.0
                        lost_cnt = 0
                if eta < 1.0 / 16.0:
                        break
        print('final result : %d / %d' % (history_accuracy, len(test_data)))

main()