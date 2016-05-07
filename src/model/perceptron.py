# -*- coding: utf-8 -*-

from __future__ import division
import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        assert(epochs > 0)

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/10
        self.w_0 = 0.0

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Here you have to implement the Perceptron Learning Algorithm
        # to change the weights of the Perceptron

        # class omega_1 : is digit 7, class omega_2: another digit

        split_view = 0

        while split_view < self.trainingSet.input.shape[0]:

            training_epoch = self.trainingSet.input[split_view:split_view+self.epochs, :]
            label_epoch = self.trainingSet.label[split_view:split_view+self.epochs]

            delta_w = np.zeros(shape=self.weight.shape)
            delta_w_0 = 0.0

            for x, label in zip(training_epoch, label_epoch):
                dot_product = np.dot(np.array(x), self.weight) + self.w_0
                g = dot_product if label == 1 else -dot_product

                if g < 0.0:
                    delta_w += x if label == 1 else -x
                    delta_w_0 += 1.0 if label == 1 else -1.0

            self.weight += self.learningRate * delta_w
            self.w_0 += self.learningRate * delta_w_0
            split_view += self.epochs

            if verbose:
                validation = self.evaluate(self.validationSet.input)
                misclassified = len([i for i, j in zip(validation, self.validationSet.label) if i != j])
                print "Validating perceptron after epoch", split_view, ", misclassified: ", misclassified,\
                    ", accurarcy:", (1-misclassified/len(validation))*100, "%"

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Here you have to implement the classification for one instance,
        # i.e., return True if the testInstance is recognized as a 7,
        # False otherwise
        return self.fire(testInstance)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input

        # Here is the map function of python - a functional programming concept
        # It applies the "classify" method to every element of "test"
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        # I already implemented it for you to see how you can work with numpy
        return Activation.sign(np.dot(np.array(input), self.weight) + self.w_0)
