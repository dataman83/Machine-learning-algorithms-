import pandas as pd
import numpy as np


car_data = pd.read_csv("car.csv", names =['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

#this function defines the entropy calculators; this calculates impurities
def entropy(tr_cl):
    elements, counts = np.unique(tr_cl, return_counts = True) #here we are taking unique values from the target column
    E = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))]) #this formula caluclates the entropy
    return E


#the following function determines the information gain

def information_gainer(data, attribute_split, name_target = "class"):
    sum_of_entropy = entropy(data[name_target])
    vals, counts = np.unique(data[attribute_split], return_counts=True)
    #now we calculate the weighted entropy
    w_E = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[attribute_split]==vals[i]).dropna()[name_target])for i in range(len(vals))])
    #now we calculate the information gain
    I_G = sum_of_entropy - w_E
    return I_G

# we now implement  the Iterative Dichotomiser 3 algorithm

def id3(data, originaldata, features, attribute_name_of_target = "class", node_parent_class =None):
    if len(np.unique(data[attribute_name_of_target])) <= 1: #when target values are all same, this will be returned
        return np.unique(data[attribute_name_of_target])[0]
    #when the dataset is empty, this will be returned
    elif len(data) == 0:
        return np.unique(originaldata[attribute_name_of_target])[np.argmax(np.unique(originaldata[attribute_name_of_target], return_counts=True)[1])]
    #when there are no features
    elif len(features) == 0:
        return node_parent_class
    #finally when no above mentioned condition is true we will grow the tree
    else:
        node_parent_class = np.unique(data[attribute_name_of_target])[np.argmax(np.unique(data[attribute_name_of_target], return_counts=True)[1])]
    #now we will select the best feature that'll split the dataset
        value_items = [information_gainer(data, features,attribute_name_of_target)for feature in features]
        optimal_feature_index = np.argmax(value_items)
        optimal_feature = features[optimal_feature_index]

        tree = {optimal_feature:{}}
    #best information gain feature is now removed
        features = [i for i in features if i!= optimal_feature]
    #tree is now grown under the root node
        for value in np.unique(data[optimal_feature]):
            value = value
            data_new = data.where(data[optimal_feature]==value).dropna()
            #the id3 alg is called
            new_tree = id3(data_new, car_data, features,attribute_name_of_target,node_parent_class)
            tree[optimal_feature][value] = new_tree
        return(tree)


def predictor(qry,tree, default=1):
    for key in list(qry.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][qry[key]]
            except:
                return default

            solution = tree[key][qry[key]]
            if isinstance(solution,dict):
                return predictor(qry, solution)
            else:
                return solution



def split_train_test(car_data):
    data_training = car_data.iloc[:60].reset_index(drop=True)
    data_testing = car_data.iloc[60:].reset_index(drop=True)
    return data_training, data_testing
data_training = split_train_test(car_data)[0]
data_testing = split_train_test(car_data)[1]


def tester(data, tree):
    queries = data.iloc[:,:-1].to_dict(orient="records")
    prediction_made = pd.DataFrame(columns=["prediction_made"])

    for i in range(len(data)):
        prediction_made.loc[i,"prediction_made"] = predictor(queries[i], tree, 1.0)

    print("prediction accuracy is: ", (np.sum(prediction_made["prediction_made"]==data["class"])/len(data))*100 , "%")


tree = id3(data_training,data_training, data_training.columns[:-1])
tester(data_testing, tree)











