from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import pandas as pd
import itertools
import numpy as np


df = pd.read_csv("../datasets/titanic.csv")
# df = pd.read_csv("../datasets/winequality-red.csv")

# X = df.drop("quality", axis=1)
# y = df["quality"]
X = df.drop("Survived", axis=1)
y = df["Survived"]

X = pd.get_dummies(X)
X.fillna(X.mean(), inplace=True)

# Define the hyperparameters to be searched
dt_param_grid = {
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, None],
    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

knn_param_grid = {
    "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
    "n_neighbors": [i for i in range(1, 21, 1)]
}

nn_param_grid = {
    "hidden_layer_sizes": [x for x in itertools.product((10, 20, 30, 40, 50, 100), repeat=3)],
    "activation": ["identity", "logistic", "tanh", "relu"]
    #  https://datascience.stackexchange.com/questions/19768/how-to-implement-pythons-mlpclassifier-with-gridsearchcv
}

svm_param_grid = {
    "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
    "C": [i for i in range(1, 11)]
}

ada_param_grid = {
    "n_estimators": [i for i in range(1, 51)],
    "learning_rate": [i for i in np.linspace(.000001, .01, 20)]
}


classifiers = {
    # "Decision Tree": (DecisionTreeClassifier(), dt_param_grid),
    # "K Nearest Neighbors": (KNeighborsClassifier(), knn_param_grid),
    # "Neural Network": (MLPClassifier(), nn_param_grid),
    "Support Vector Machine": (SVC(), svm_param_grid),
    # "Adaboost": (AdaBoostClassifier(), ada_param_grid)
}


for classifier in classifiers:
    print(classifier)
    classifier_algo = classifiers[classifier][0]
    classifier_param_grid = classifiers[classifier][1]
    grid_search = GridSearchCV(classifier_algo, classifier_param_grid, cv=5, scoring="f1", n_jobs=-1)

    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_params["Name"] = classifier

    best = grid_search.best_estimator_
    # df = df.append(pd.DataFrame(best_params, index=[0]), ignore_index=True)
    print(best_params)
