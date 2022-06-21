import numpy as np
import pandas as pd


data_test = pd.read_csv('fashion-mnist_test.csv')
data_train = pd.read_csv('fashion-mnist_train.csv')

data_train = np.array(data_train)
data_test = np.array(data_test)


#we'll start by defining the size of each layer
size_input = 784 # 28 x 28, hence for each pixel
size_hidden = 30
size_output = 10 # to pick from 10 label predictions

# the shape of the weight matrices are as follows
weight1_shape = (size_output, size_hidden)
weight2_shape = (size_hidden, size_output)

#when starting we give random value to weights using python's seed function
weight1 = np.random.uniform(-1,1,weight1_shape)/np.sqrt(size_input)
weight2 = np.random.uniform(-1,1,weight2_shape)/np.sqrt(size_hidden)

#we'll now initialize the biases
bias_1 = np.full(size_hidden, 0.)
bias_2 = np.full(size_output, 0.)

#setting the parameters for training
number_of_iterations = 30
size_of_batch = 20
learning_rate = 3

#number of test images to use
test_number = 10000

#we'll start defining our forward propagation function
def f_propagation(picture):
    x = picture.flatten() / 255 #gray scale images contain max value of 255, so by dividing each element with 255 we get numbers less than or equal to 1
    # lets activate our hidden layer with sigmoid function
    hidden_layer = np.dot(x,weight1) + bias_1
    activated_hidden_layer = 1/(1+np.exp(-hidden_layer))
    #let's activate our output layer now with softmax function
    output_layer = np.dot(activated_hidden_layer, weight2) + bias_2
    output_layer_E = np.exp(output_layer)
    activated_output_layer = output_layer_E/ output_layer_E.sum()
    return x, hidden_layer, activated_hidden_layer, output_layer, activated_output_layer

#the following function calculates the loss
def loss_calculator(activated_output_layer, t):
    # this is cross-entropy loss for a given output and target number t
    return -np.log(activated_output_layer[t])

#the next function defines the backpropagation process
def backpropagation(x, hidden_layer, activated_hidden_layer, activated_output_layer, t) :
    #the following are derivatives d_y of the loss with respect to each parameter y
    d_bias_2 = activated_output_layer
    d_bias_2[t] -= 1
    d_weight_2 = np.outer(activated_hidden_layer, d_bias_2)
    d_bias_1 = np.dot(weight2, d_bias_2) * activated_hidden_layer * (1-activated_hidden_layer)
    d_weight1 = np.outer(x, d_bias_1)
    return d_weight1, d_weight_2, d_bias_1, d_bias_2

#now we will define the training function that adjusts the weights and biases
def trainer():
     for k in range(number_of_iterations):
         #we'll start by initizlizing the derivatives for the batch
         d_weight1_sum = np.zero(weight1_shape)
         d_weight2_sum = np.zeros(weight2_shape)
         d_bias_1_sum = np.zeros(size_hidden)
         d_bias_2_sum = np.zeros(size_output)
         # here we'll use the index for the training pic and label
         for i in range(size_of_batch):
             index = k * size_of_batch + i
             picture = data_test[index]


             x, hidden_layer, activated_hidden_layer, output_layer, activated_output_layer = f_propagation(picture)

             d_weight1, d_weight2, d_bias_1, d_bias_2 = backpropagation(x,hidden_layer, activated_hidden_layer, output_layer, activated_output_layer, t)

             d_weight1_sum += d_weight1
             d_weight2_sum += d_weight2
             d_bias_1_sum += d_bias_1
             d_bias_2_sum += d_bias_2

             #we can finally update the weights and biases
         weight1[:] -= learning_rate * d_weight1_sum
         weight2[:] -= learning_rate * d_weight2_sum
         bias_1 [:] -= learning_rate * d_bias_1_sum
         bias_2 [:] -= learning_rate * d_bias_2_sum

         # the [:] notation is used to modify the weights and biases, because without it they are considered undefined local variables

def tester():
    #this function takes a random picture from the dataset and checks if they're the same as the one provided by nn
    randomiser = np.random.randint(0, len(data_test))
    picture = data_test[randomiser]
    label = data_test[randomiser]
    x, hidden_layer, activated_hidden_layer, output_layer, activated_output_layer = f_propagation(picture)
    result = activated_output_layer.argmax()
    return result == label

def accuracy_checker():

    accuracy = 0
    for i in range(test_number):
        if tester():
            accuracy += 1
    return accuracy/ test_number

accuracy_checker()















