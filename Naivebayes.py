import numpy as np
from collections import Counter, defaultdict
import pandas as pd
import statistics as st

def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob


def naive_bayes(training, outcome):
    classes = np.unique(outcome)
    rows, cols = np.shape(training)
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)

    class_probabilities = occurrences(outcome)

    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset = training[row_indices, :]
        r, c = np.shape(subset)
        for j in range(0, c):
            likelihoods[cls][j] += list(subset[:, j])

    for cls in classes:
        for j in range(0, cols):
            likelihoods[cls][j] = occurrences(likelihoods[cls][j])
    return likelihoods, class_probabilities, classes

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def cross_validation_split(dataset):
    dataset_split = list()
    dataset_copy = dataset.copy()
    fold_size = int(len(dataset) / 5)
    for i in range(5):
        f = pd.DataFrame(columns=['clump','cell_size','cell_shape','adhesion','epithelial','bare_nuclei','chromatin','nucleoli','mitoses','output_cancer'])
        f=f.append(dataset_copy.head(fold_size))
        dataset_copy=dataset_copy.iloc[fold_size:].reset_index(drop=True)
        dataset_split.append(f)
    return dataset_split

def evaluate_algorithm(dataset):
    folds = cross_validation_split(dataset)
    scores = list()
    for i in range(len(folds)):
        f1 = folds.copy()
        test_set = f1[i]
        f1.pop(i)
        train_set = pd.concat(f1)
        x = np.array(train_set.drop('output_cancer', axis=1).copy())
        y = np.array(train_set['output_cancer'].copy())
        likelihoods, class_probabilities,classes = naive_bayes(x, y)
        prediction = list()
        new_sample = np.array(test_set.drop('output_cancer', axis=1).copy())
        for new in new_sample:
            results = {}
            for cls in classes:
                class_probability = class_probabilities[cls]
                for i in range(0, len(new)):
                    relative_values = likelihoods[cls][i]
                    if new[i] in relative_values.keys():
                        class_probability *= relative_values[new[i]]
                    else:
                        class_probability *= 0
                results[cls] = class_probability
            prediction.append(max(results,key=results.get))
        actual = np.array(test_set['output_cancer'].copy())
        accuracy = accuracy_metric(actual, prediction)
        scores.append(accuracy)
    return scores

df = pd.read_csv(r'cancer.csv',header=None)

# Removed the first attribute “Sample code number” as this is having unique entries which will not be used for forming any patterns.
df = df.drop(df.columns[0], axis=1)

df.columns=['clump','cell_size','cell_shape','adhesion','epithelial','bare_nuclei','chromatin','nucleoli','mitoses','output_cancer']

change = float('nan')
for col in list(df):
    df[col] = pd.to_numeric(df[col],errors='coerce',downcast='float')
    df = df.replace(change, int(df[col].mode()[0]))

final_accuracy=list()

# performing 10 times five fold
for i in range(10):
    a = evaluate_algorithm(df)
    df = df.sample(frac=1).reset_index(drop=True)
    final_accuracy += a
print(final_accuracy)
print(st.mean(final_accuracy))
print(st.pstdev(final_accuracy))
