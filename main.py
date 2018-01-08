from numpy import exp, array, random, dot, genfromtxt, argmax
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

inputs = 4
outputs = 7
input_div = 10000

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((inputs, outputs)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    def think_print(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        #print "inputs"
        #print inputs
        #print "weights"
        #print self.synaptic_weights
        #print "dot product"
        #print dot(inputs, self.synaptic_weights)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    file_data = genfromtxt('color_train.csv', delimiter=',')
    file_data_test = genfromtxt('color_test.csv', delimiter=',')
    #print "Input"
    file_input = file_data[:,:inputs]
    file_input = file_input / input_div
    print file_input


    ax.scatter(file_input[:18,0], file_input[:18,1], file_input[:18,2], c='b', marker='o')
    ax.scatter(file_input[19:36,0], file_input[19:36,1], file_input[19:36,2], c='g', marker='x')
    ax.scatter(file_input[37:54,0], file_input[37:54,1], file_input[37:54,2], c='r', marker='x')
    ax.scatter(file_input[53:72,0], file_input[53:72,1], file_input[53:72,2], c='c', marker='x')
    ax.scatter(file_input[73:90,0], file_input[73:90,1], file_input[73:90,2], c='m', marker='x')
    ax.scatter(file_input[91:108,0], file_input[91:108,1], file_input[91:108,2], c='y', marker='x')
    ax.scatter(file_input[109:126,0], file_input[109:126,1], file_input[19:36,2], c='k', marker='x')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    #print "Output"
    file_output = file_data[:,inputs:]
    #print file_output

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    #training_set_inputs = array([[0.0259,0.0245,0.0097,0.0472], [0.0265,0.0249,0.0098,0.0479], [0.5876,0.5716,0.1595,0.7859], [0.2171,0.3126,0.0889,0.4026]])
    #training_set_outputs = array([[1,1,0,0], [0,0,1,0], [0, 0, 0, 1]]).T
    training_set_inputs = file_input
    training_set_outputs = file_output

    #print "Inputs: "
    #print training_set_inputs

    #print "Outputs: "
    #print training_set_outputs

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

	
    # Test the neural network with a new situation.
    file_input_test = file_data_test[:,:inputs]
    file_output_test = file_data_test[:,inputs:]
    for i in range(len(file_input_test)):
        #print "Considering new situation "
        #print file_input_test[i] / input_div
        #print "Real"
        realIndex = argmax(file_output_test[i])
        print file_output_test[i]	
        #print "Calculated"
        testIndex = argmax(neural_network.think_print(array(file_input_test[i] / input_div)))
        if (realIndex != testIndex):
            print "Error"
        print neural_network.think_print(array(file_input_test[i] / input_div))

    print file_input_test[:,0]
    #print neural_network.think_print(array([2513,3811,1037,4761]))
