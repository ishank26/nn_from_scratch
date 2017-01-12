# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import time


class neural_net(object):

    def __init__(self, layers_dim):
        # [input_ dim, hidden_dim, output_dim]
        input_dim = layers_dim[0]
        hidden_dim = layers_dim[1]
        output_dim = layers_dim[2]

        # layer 1 === input---> hidden === (W1,b1)
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        # layer 2 === hidden---> hidden === (W2,b2)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))
        # layer 3 === hidden---> output === (W3,b3)
        self.W3 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b3 = np.zeros((1, output_dim))

    def softmax(self, input):
        # Generate probabilties
        probs = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        return probs

    def tanh(self, input):
        act_op = np.tanh(input)
        return act_op

    def fprop(self, X):
        # Forward prop
        input = X
        pre_act1 = X.dot(self.W1) + self.b1
        act1 = np.tanh(pre_act1)
        pre_act2 = act1.dot(self.W2) + self.b2
        act2 = np.tanh(pre_act2)
        pre_act3 = act2.dot(self.W3) + self.b3
        act3 = self.softmax(pre_act3)
        return pre_act1, pre_act2, pre_act3, act1, act2, act3

    def predict(self, X):
        pre_act1, pre_act2, pre_act3, act1, act2, act3 = self.fprop(X)
        # Get max prob. for class
        return np.argmax(act3, axis=1)

    def crossentropy_loss(self, X, y, reg_lambda):
        num_data = len(X)
        pre_act1, pre_act2, pre_act3, act1, act2, act3 = self.fprop(
            X)  # get probs

        # Log probabilities for crossentropy
        logprob = -np.log(act3[range(num_data), y])
        # Summation of all log probabilites
        loss = np.sum(logprob)

        # L2 weight decay for regularization = lambda * sum(theta**2)
        loss += reg_lambda / 2 * (np.sum(np.square(self.W1)) +
                                  np.sum(np.square(self.W2)) + np.sum(np.square(self.W3)))
        return 1.0 / num_data * loss

    def bprop(self, X, y, reg_lambda, epsilon):
        # Back prop
        num_data = len(X)
        pre_act1, pre_act2, pre_act3, act1, act2, act3 = self.fprop(X)
        # error signal 4
        esg4 = act3
        esg4[range(num_data), y] -= 1  # δ4
        # error signal 3
        esg3 = esg4.dot(self.W3.T) * (1 - np.square(act2))  # δ3
        # error signal 2
        esg2 = esg3.dot(self.W2.T) * (1 - np.square(act1))  # δ2

        # Partial derivatives of params
        dW3 = act2.T.dot(esg4)
        db3 = np.sum(esg4, axis=0, keepdims=True)
        dW2 = act1.T.dot(esg3)
        db2 = np.sum(esg3, axis=0, keepdims=True)
        dW1 = X.T.dot(esg2)
        db1 = np.sum(esg2, axis=0, keepdims=True)
        # print dW3,"\n",dW2,"\n",dW1,"\n"
        # print db3,"\n",db2,"\n",db1,"\n"

        # Regularization of weights
        dW3 += reg_lambda * self.W3
        dW2 += reg_lambda * self.W2
        dW1 += reg_lambda * self.W1

        # Gradient descent for whole batch
        self.W3 += -epsilon * dW3
        self.b3 += -epsilon * db3
        self.W2 += -epsilon * dW2
        self.b2 += -epsilon * db2
        self.W1 += -epsilon * dW1
        self.b1 += -epsilon * db1

        return self.W3, self.b3, self.W2, self.b2, self.W1, self.b1

    def train(self, X, y, reg_lambda, epsilon, num_pass=25000, print_loss=True):
        # Train for each iteration
        for i in xrange(0, num_pass):
            self.W3, self.b3, self.W2, self.b2, self.W1, self.b1 = self.bprop(
                X, y, reg_lambda, epsilon)
            if print_loss and i % 1000 == 0:
                print "Loss after iteration %i: %f" % (i, self.crossentropy_loss(X, y, reg_lambda))

        # Make params. dict.
        weights = ["W1", "W2", "W3"]
        num_W = [self.W1, self.W3, self.W3]
        biases = ["b1", "b2", "b3"]
        num_b = [self.b1, self.b2, self.b3]

        nn_model = dict(zip(weights, num_W))
        nn_model.update(dict(zip(biases, num_b)))

        return nn_model

    def visualize_preds(self, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
        y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7
        # Space between grid points
        h = 0.005
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        # Predict values for whole gid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        # print Z.shape
        Z = Z.reshape(xx.shape)
        # Plot the contour
        plt.figure(figsize=(7, 5))
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        plt.show()

    def animate_preds(self, X, y, reg_lambda, epsilon, num_pass=35000):
        # Train neural network and animate the training procedure
        i = 0
        plt.ion()
        plt.figure(figsize=(8, 6))
        while i < num_pass:
            self.W3, self.b3, self.W2, self.b2, self.W1, self.b1 = self.bprop(
                X, y, reg_lambda, epsilon)
            x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
            y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7
            h = 0.005
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            # make predictions for all grid points
            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.gca().cla()  # get current axes and clear for plot new contour
            plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
            display.clear_output(wait=True)
            display.display(plt.gcf())
            # plt.draw() # not working in matplotlib
            i += 1
            time.sleep(0.8)
