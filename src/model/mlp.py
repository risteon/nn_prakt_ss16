
import numpy as np
from sklearn.metrics import accuracy_score

from util.activation_functions import Activation
from util.loss_functions import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 cost='crossentropy', learning_rate=0.01, epochs=50):

        """
        A MNIST recognizer

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.classification_task = True if output_task == 'classification' else False # Either classification or regression
        self.output_activation = output_activation
        self.output_activation_func = Activation.get_activation(self.output_activation)
        self.cost = cost

        if self.cost == 'crossentropy':
            self.cost_function = CrossEntropyError()
        else:
            # nothing else supported...
            raise ValueError('not supported')

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.input_weights = input_weights

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """

        outp = inp
        first = True
        for layer in self.layers:
            if first:
                first = False
            else:
                # add bias one (TODO: this behavior of the logistic layer is quite poor)
                outp = np.insert(outp, 0, 1)

            outp = layer.forward(outp)

        # apply softmax
        if self.classification_task:
            self.outp = self.output_activation_func(outp)

        return self.outp

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        Error of output using cost function (scalar value)
        """
        return self.cost_function.calculate_error(target, self._get_output_layer().outp)

    def _update_weights(self, label):
        """
        Update the weights of the layers by propagating back the error
        """

        output_size = self._get_output_layer().n_out
        # create one-hot target output
        target_outp = np.asmatrix(np.zeros(output_size))
        target_outp[0, label] = 1.0
        # create dummy next weights as vector of ones
        next_weights = np.asmatrix(np.ones((output_size, output_size)))

        output_layer = True
        for layer in reversed(self.layers):
            if output_layer:
                # softmax layer derivatives
                # do I really need to apply softmax prime? Slide 45 in Backpropagation says NO
                # next_derivatives = Activation.softmax_prime(target_outp - self.outp)
                next_derivatives = target_outp - self.outp

                layer.computeDerivative(next_derivatives,
                                        next_weights)
                output_layer = False
            else:
                layer.computeDerivative(next_layer.deltas, np.transpose(next_layer.weights))

            layer.updateWeights(self.learning_rate)
            next_layer = layer

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validation_set.label,
                                          self.evaluate(self.validation_set))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.training_set.input,
                              self.training_set.label):

            # Use LogisticLayer to do the job
            # Feed it with inputs

            # Do a forward pass to calculate the output and the error
            self._feed_forward(img)

            # Compute the derivatives w.r.t to the error,
            # Update weights in the online learning fashion
            self._update_weights(label)

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        sm_output = self._feed_forward(test_instance)
        return np.argmax(sm_output)

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
            test = self.test_set.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
