import math
import numpy as np
import pandas as pd
import statistics as st
from math import sqrt
from random import randrange

def cross_validation_split(dataset):
    #Doing cross validation for the dataset
    dataset_split = list()
    dataset_copy = dataset.copy()
    fold_size = int(len(dataset) / 5)
    for i in range(5):
        f = pd.DataFrame(columns=['clump','cell_size','cell_shape','adhesion','epithelial','bare_nuclei','chromatin','nucleoli','mitoses','output_cancer'])
        f=f.append(dataset_copy.head(fold_size))
        dataset_copy=dataset_copy.iloc[fold_size:].reset_index(drop=True)
        dataset_split.append(f)
    return dataset_split

def predict(tree, row):
    # Predicting the output for test data
    for x in tree.childs:
        if row[tree.value] == x.value:
            if x.next.childs:
                return predict(x.next,row)
            else:
                return x.next.value

def accuracy_metric(actual, predicted):
    # Calculating the accuracy
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def predict_bag_result(trees, row):
    pd=list()
    for tree in trees:
        pd.append(predict(tree,row))
    return max(set(pd), key=pd.count)

def subdata(dataset):
    sample = pd.DataFrame(columns=['clump','cell_size','cell_shape','adhesion','epithelial','bare_nuclei','chromatin','nucleoli','mitoses','output_cancer'])
    sample_of_n = len(dataset)
    while len(sample) < sample_of_n:
        index = randrange(len(dataset))
        sample=sample.append(dataset.iloc[index],ignore_index=True)
    return sample

def random_forest(train, test):
    trees = list()
    number_of_trees = 10
    for i in range(number_of_trees):
        sample = subdata(train)
        features = list(sample.drop('output_cancer', axis=1).copy())
        sub_feature = int(sqrt(len(features)))
        for f in range(sub_feature):
            index = randrange(len(list(sample))-1)
            sample = sample.drop(sample.columns[index],axis=1)
        v1 = np.array(sample.drop('output_cancer', axis=1).copy())
        v2 = np.array(sample['output_cancer'].copy())
        new_features=list(sample.drop('output_cancer', axis=1).copy())
        tree_clf = DecisionTreeClassifier(x=v1, feature_names=new_features, labels=v2)
        tree = tree_clf.id3()
        trees.append(tree)
    predictions = [predict_bag_result(trees, row) for index, row in test.iterrows()]
    return(predictions)

def evaluate_algorithm(dataset):
    # The ID3 alg is evaluated from here
    folds = cross_validation_split(dataset)
    scores = list()
    for i in range(len(folds)):
        f1 = folds.copy()
        test_set = f1[i]
        f1.pop(i)
        train_set = pd.concat(f1)
        prediction = random_forest(train_set, test_set)
        correct = [row['output_cancer'] for index, row in test_set.iterrows()]
        acc = accuracy_metric(correct, prediction)
        scores.append(acc)
    return scores

#contains information on nodes of the tree
class Node:

    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None

# This is the ID3 classifier
class DecisionTreeClassifier:

    def __init__(self, X, feature_names, labels):
        self.X = X
        self.feature_names = feature_names
        self.labels = labels
        self.labelCategories = list(set(labels))
        self.labelCategoriesCount = [list(labels).count(x) for x in self.labelCategories]
        self.node = None
        self.entropy = self._get_entropy([x for x in range(len(self.labels))])

    def _get_entropy(self, x_ids):
        # This function calculates the entropy

        labels = [self.labels[i] for i in x_ids]
        label_count = [labels.count(x) for x in self.labelCategories]
        entropy = sum([-count / len(x_ids) * math.log(count / len(x_ids), 2) if count else 0 for count in label_count])
        return entropy

    def _get_information_gain(self, x_ids, feature_id):
        #This function calculates the information gain

        info_gain = self._get_entropy(x_ids)
        x_features = [self.X[x][feature_id] for x in x_ids]
        feature_vals = list(set(x_features))
        feature_vals_count = [x_features.count(x) for x in feature_vals]
        feature_vals_id = [
            [x_ids[i]
             for i, x in enumerate(x_features)
             if x == y]
            for y in feature_vals]

        info_gain = info_gain - sum([val_counts / len(x_ids) * self._get_entropy(val_ids)
                                     for val_counts, val_ids in zip(feature_vals_count, feature_vals_id)])
        return info_gain

    def _get_feature_max_information_gain(self, x_ids, feature_ids):
        #This function gets the attribute with max information gain

        features_entropy = [self._get_information_gain(x_ids, feature_id) for feature_id in feature_ids]
        # find the feature that maximises the information gain
        max_id = feature_ids[features_entropy.index(max(features_entropy))]
        return self.feature_names[max_id], max_id

    def id3(self):
        #Initializing the ID3 classifier
        x_ids = [x for x in range(len(self.X))]
        feature_ids = [x for x in range(len(self.feature_names))]
        self.node = self._id3_recv(x_ids, feature_ids, self.node)
        return self.node

    def _id3_recv(self, x_ids, feature_ids, node):
        #constructs the tree
        if not node:
            node = Node()  # initialize nodes
        labels_in_features = [self.labels[x] for x in x_ids]
        if len(set(labels_in_features)) == 1:
            node.value = self.labels[x_ids[0]]
            return node
        if len(feature_ids) == 0:
            node.value = max(set(labels_in_features), key=labels_in_features.count)  # compute mode
            return node
        best_feature_name, best_feature_id = self._get_feature_max_information_gain(x_ids, feature_ids)
        node.value = best_feature_name
        node.childs = []
        feature_values = list(set([self.X[x][best_feature_id] for x in x_ids]))
        for value in feature_values:
            child = Node()
            child.value = value
            node.childs.append(child)
            child_x_ids = [x for x in x_ids if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                child.next = max(set(labels_in_features), key=labels_in_features.count)
                print('')
            else:
                if feature_ids and best_feature_id in feature_ids:
                    to_remove = feature_ids.index(best_feature_id)
                    feature_ids.pop(to_remove)
                child.next = self._id3_recv(child_x_ids, feature_ids, child.next)
        return node

df = pd.read_csv(r'cancer.csv',header=None)

# Removed the first attribute “Sample code number” as this is having unique entries which will not be used for forming any patterns.
df = df.drop(df.columns[0], axis=1)

df.columns=['clump','cell_size','cell_shape','adhesion','epithelial','bare_nuclei','chromatin','nucleoli','mitoses','output_cancer']

# dataset has missing instances so replaced those instances with most repetitive instance from that attribute
for col in list(df):
    df = df.replace('?', df[col].mode()[0])

final_accuracy=list()

# performing 10 times five fold
for i in range(10):
    a = evaluate_algorithm(df)
    df = df.sample(frac=1).reset_index(drop=True)
    final_accuracy += a
print(final_accuracy)
print(st.mean(final_accuracy))
print(st.pstdev(final_accuracy))
