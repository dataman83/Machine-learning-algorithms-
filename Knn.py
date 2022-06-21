import numpy as np
import pandas as pd
import statistics

class KNN():

    def __init__(self, distance_value): #this function will inititate the whole Knn_algorithm
        self.distance_metrics = distance_value

    # the following function measures the Euclidean or Manhattan distance between two wines based on attribute

    def distance_calculator(self, coordinate_training_data, coordinate_test_data):
        if(self.distance_value == "EUC"): # EUC denotes euclidean distance
            distance == 0
            for i in range(len(coordinate_training_data) -1):
                distance = distance + (coordinate_training_data[i] - coordinate_test_data[i])**2#Basically first step of calculating Pythagorean distance between two points
            EUC_distance = np.sqrt(distance)#following from the first step, we square root the number using numpy's sqr function
            return EUC_distance
        elif(self.distance_value == "MANH"):
            distance == 0
            for i in range(len(coordinate_training_data) -1):
                distance = distance + abs(coordinate_training_data[i] - coordinate_test_data[i]) #Manhattan distance is the absolute difference between two points on a data set; hence the abs function
            MANH_distance = distance
            return MANH_distance

    def near_neighbours(self, X_train, data_test, k):

        list_of_distances = []# we are creating an empty list to accomodate our 'E' or 'M' distances of the training data set and test data

        for data_training in X_train:
            distance = self.distance_calculator(data_training, data_test)
            list_of_distances.append(data_training, distance)
        list_of_distances.sort(key=lambda x:x[1])# we need to sort the distance metrics in ascending order hence the lambda key, since we just need distances hence index 0
        list_of_neighbours = []  # an empty list to store k_nearest neighbour
        for j in range(k):
            list_of_neighbours.append(list_of_distances[j][0]) # now we only need the corresponding labels and not distances
        return list_of_neighbours


    def predictor(self, X_train, data_test, k):

        neighbours = self.near_neighbours(X_train, data_test, k)

        for data in neighbours:
            list_of_labels = [] # An empty list called class is created to store all the near neighbour data
            list_of_labels.append(data[-1])
        class_of_prediction = statistics.mode(list_of_labels)#from our statistics library we are calling the mode function to find which class occurs the highest number of times

        return class_of_prediction


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine_data = pd.read_csv("wine.csv")

X = wine_data.drop(columns='Wine', axis = 1)
Y = wine_data['Wine']
X = X.to_numpy()
Y = Y.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=123)
X_train = np.insert(X_train, 13, Y_train, axis=1)

classifier = KNN(distance_value='EUC')
prediction = classifier.predictor(X_train, X_test[0], k=5)









