from math import sqrt
import statistics as st
import pandas as pd
import numpy as np

def cross_validation(dataset):
    dt_split = list()
    dt_copy = dataset.copy()
    fold_size = int(len(dataset) / 5)
    for i in range(5):
        features = pd.DataFrame(columns=['cthick','csize','cshape','adhesion','ecsize','bnuclei','bchromatin','nucleoli','mitoses','class'])
        features=features.append(dt_copy.head(fold_size))
        dt_copy=dt_copy.iloc[fold_size:].reset_index(drop=True)
        dt_split.append(features)
    return dt_split


# Calculate accuracy percentage
def getting_accuracy(actual, predicted):
    initial = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            initial += 1
    return initial / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def running_algorithm(dataset):
    fold_value = cross_validation(dataset)
    val = list()
    for i in range(len(fold_value)):
        val1 = fold_value.copy()
        test_data = val1[i]
        val1.pop(i)
        train_data = pd.concat(val1)
        prediction = k_nearest_neighbors(train_data, test_data)
        correct = [row['class'] for index, row in test_data.iterrows()]
        accuracy = getting_accuracy(correct, prediction)
        val.append(accuracy)
    return val

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate the most similar neighbors
def neighbors_get(train, test_row, num_neighbors):
    distances = list()
    for index,train_row in train.iterrows():
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = pd.DataFrame(columns=['cthick','csize','cshape','adhesion','ecsize','bnuclei','bchromatin','nucleoli','mitoses','class'])
    for i in range(num_neighbors):
        neighbors = neighbors.append(distances[i][0],ignore_index=True)
    return neighbors


# Make a prediction with neighbors
def predict_output(train, test_row, num_neighbors):
    neighbors = neighbors_get(train, test_row, num_neighbors)
    output_values = [row['class'] for index,row in neighbors.iterrows()]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test):
    predictions = list()
    num_neighbors = 5
    for index,row in test.iterrows():
        output = predict_output(train, row, num_neighbors)
        predictions.append(output)
    return(predictions)

def preprocessing(data_frame):
    change = float('nan')
    for col in list(data_frame):
        data_frame[col] = pd.to_numeric(data_frame[col], errors='coerce', downcast='float')
        data_frame = data_frame.replace(change, int(data_frame[col].mode()[0]))
    new_copy = data_frame.copy()

    for col in list(data_frame):
        for i in range(len(data_frame)):
            value = (data_frame[col][i] - data_frame[col].min()) / (data_frame[col].max() - data_frame[col].min())
            new_copy[col][i] = value
    data_frame = new_copy.copy()
    return  data_frame

def result(data_frame):
    acc = list()
    for i in range(10):
        val = running_algorithm(data_frame)
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)
        acc += val
    #print(acc)
    print(st.mean(acc))
    print(st.pstdev(acc))


data_frame = pd.read_csv(r'cancer.csv', header=None)

# Removed the first attribute “Sample code number” as this is having unique entries which will not be used for forming any patterns.
data_frame = data_frame.drop(data_frame.columns[0], axis=1)

data_frame.columns = ['cthick', 'csize', 'cshape', 'adhesion', 'ecsize', 'bnuclei', 'bchromatin', 'nucleoli', 'mitoses', 'class']
pre = preprocessing(data_frame)
result(pre)

