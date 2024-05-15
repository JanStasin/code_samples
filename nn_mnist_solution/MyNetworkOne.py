import numpy as np


def sigmoid(Z, derivative=False):
    if derivative == True:
        return sigmoid(Z)*(1-sigmoid(Z))
    return 1.0 / (1.0 + (np.exp(-Z)))

def ReLu(Z, derivative=False):
    if derivative == True:
        return Z > 0
    return np.maximum(0,Z)

def SoftMax(Z):
    '''computes softmax takes Z as an argument and returns probabilities '''
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)
    
def vectorize_labels(labels):
    """Converts the labels into into a vectors with a 1.0 in the respective label
    position and zeroes elsewhere.  This is used further in cost function computations."""
    v = np.zeros((labels.size, max(labels) + 1))
    v[np.arange(labels.size), labels] = 1.0
    v = v.T
    return v


class MyNetworkOne(object):
    
    def __init__(self ,L_sizes):
        '''network setup based on layer sizes array supplied at the beginning
        each element of the array defines the number of nodes for each layer.
        784 is the basic input size 10 is the basic output size (last element of the array)
        we dont need to create activation values in advance they will be dynamically computed during feedforward 
        procedure '''
        ## initializing weights and biases at each layer
        self.W1 = np.random.rand(L_sizes[1], L_sizes[0])- 0.5
        print('W1 shape: ', self.W1.shape)
        self.W2 = np.random.rand(L_sizes[2], L_sizes[1])- 0.5
        print('W2 shape: ', self.W2.shape)
        self.W3 = np.random.rand(L_sizes[3], L_sizes[2]) - 0.5
        print('W3 shape: ', self.W3.shape)
        self.B1 = np.random.rand(L_sizes[1],1)- 0.5
        print('B1 shape: ', self.B1.shape)
        self.B2 = np.random.rand(L_sizes[2],1)- 0.5
        print('B2 shape: ', self.B2.shape)
        self.B3 = np.random.rand(L_sizes[3],1)- 0.5
        print('B3 shape: ', self.B3.shape)
    
    
    def forward_feed(self, data_input, a_func): ### can activation function be added dynamically?
        '''runs the feed forward through the whole network
        inputs: network state, input data and activation function
        outputsL: zs and activations for each layer'''
        z1= np.matmul(self.W1, data_input) + self.B1 ## dot product of input matrix and weights matrix plus respective biase
        A1= a_func(z1) ## application of activation function
        
        z2= np.matmul(self.W2, A1) + self.B2 ## repeat for the next layers
        A2= a_func(z2)
        
        z3= np.matmul(self.W3, A2) + self.B3
        #A3= sigmoid(z3)
        A3= SoftMax(z3)
        #print(A3)
        return z1, z2, z3, A1, A2, A3;

    def backprop_feed(self, z1, z2, z3, A1, A2, A3, data_input, labels, a_func):
        '''runs the backpropagation through the whole network
        inputs: network state, input data, and corresponding labels + activation function
        outputsL: network state, changes to Weights and Biases for each layer'''
        v_labels = vectorize_labels(labels)
        
        #output layer
        d_z3 = A3 - v_labels #activation of the output layer - the vectorized label
        d_W3 = 1/v_labels.size * np.matmul(d_z3, A2.T) #dot prod of z3 and activations on the hidden layer 2 
        d_B3 = 1/v_labels.size * np.sum(d_z3, 1)

        # hidden layer 2:
        d_z2 = np.dot(self.W3.T, d_z3) * a_func(z2, derivative = True)
        d_W2 = 1/v_labels.size * np.matmul(d_z2, A1.T) #dot prod of z2 and activations on the hidden layer 1 
        d_B2 = 1/v_labels.size * np.sum(d_z2, 1)

        # hidden layer 1:
        d_z1 = np.dot(self.W2.T, d_z2) * a_func(z1, derivative = True)
        d_W1 = 1/v_labels.size * np.matmul(d_z1, data_input.T) #dot prod of z2 and activations on the input layer 
        d_B1 = 1/v_labels.size * np.sum(d_z1, 1)
        return d_W1, d_B1, d_W2, d_B2, d_W3, d_B3 ### returning changes to weights and biases for updating
    
    def update(self, d_W1, d_B1, d_W2, d_B2, d_W3, d_B3, learning_rate):
        '''This method updates the weights and biases and applies learning rate
        Arguments: derivatives of weights and biases from backprop feed and learning rate'''
        self.W1 = self.W1 - (learning_rate * d_W1)
        self.B1 = self.B1 - (learning_rate * np.reshape(d_B1, (len(d_B1),1)))
        self.W2 = self.W2 - (learning_rate * d_W2)
        self.B2 = self.B2 - (learning_rate * np.reshape(d_B2, (len(d_B2),1)))
        self.W3 = self.W3 - (learning_rate * d_W3)
        self.B3 = self.B3 - (learning_rate * np.reshape(d_B3, (len(d_B3),1)))
        return self.W1, self.B1, self.W2, self.B2, self.W3, self.B3
   
    def interpret_output(self, output_layer):
        '''takes A3 - activations of the ouput layer and outputs the corresponding 0-9 integer'''
        return np.argmax(output_layer, 0)
    
    def get_network_accuracy(self, guesses, labels):
        '''as advertised: compares all network guesses against the corresponding labels
        Arguments: guesses and labels 
        Output: Percentage correct - accuracy''' 
        print('guesses: ', guesses)
        print('labels : ', labels)
        return np.sum(guesses == labels) / labels.size
    

    def raw_gradient_descent(self, data_input, labels, a_func, learning_rate, iterations):
        '''Runs a gradient decent algorythm on the whole input data set for the specified number of times
        Arguments: data, corresponding labels, activation function
        learning rate,  number of times to run the data through the algorythm
        Output: state of the network'''
        for i in range(iterations):
            z1, z2, z3, A1, A2, A3 = self.forward_feed(data_input, a_func)
            d_W1, d_B1, d_W2, d_B2, d_W3, d_B3 = self.backprop_feed(z1, z2, z3, A1, A2, A3,
                                                                    data_input, labels, a_func)
            self.W1, self.B1, self.W2, self.B2, self.W3, self.B3 = self.update(d_W1, d_B1, d_W2, d_B2,
                                                                               d_W3, d_B3, learning_rate)
            if i % 100 == 0:
                print("Iteration: ", i)
                guesses = self.interpret_output(A3)
                #print('first A3', A3 )
                #print('Accuracy: ', self.get_network_accuracy(guesses, labels)) 
        print('Training accuracy: ', self.get_network_accuracy(guesses, labels)) 
        #return self.W1, self.B1, self.W2, self.B2, self.W3, self.B3
        return self.get_network_accuracy(guesses, labels)

    
    def make_predictions(self, data_input, a_func):
        '''function used for getting the predictions on the validation / testing datasets
        applies interpret output function to the output layer after a forward feed with data
        Arguments: data and activation function (used for training)'''
        *_ , self.A3 = self.forward_feed(data_input, a_func)
        predictions = self.interpret_output(self.A3)
        return predictions

    def test_prediction(self, index, a_func):
        current_image = train_data[:, index, None]
        prediction = self.make_predictions(train_data[:, index, None], a_func)
        label = train_labels[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show() 