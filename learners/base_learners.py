import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


def test_plot(X, y, algo, name, dataset_name):
    train_accuracy = []
    test_accuracy = []
    train_f1 = []
    test_f1 = []

    X_rows, X_cols = X.shape

    splits = 5 / 100  # number of times we want to slice the range

    start = X_rows * splits

    num_samples_range = [int(i) for i in np.linspace(start, X_rows, 20)]

    for _, j in enumerate(num_samples_range):
        f1_train = []
        f1_test = []

        accuracy_list_train = []
        accuracy_list_test = []

        folds = StratifiedKFold(n_splits=5)

        # the mini x train and y trains (44 rows)
        X_train = X[:j].values
        y_train = y[:j]

        split_folds = folds.split(X_train, y_train)

        for a, b in split_folds:
            algo.fit(X_train[a], y_train[a])

            y_train_predictions = algo.predict(X_train[a])
            y_test_predictions = algo.predict(X_train[b])

            accuracy_list_train.append(np.sum(y_train_predictions == y_train[a]) / len(y_train[a]))
            accuracy_list_test.append(np.sum(y_test_predictions == y_train[b]) / len(y_train[b]))

            f1_train_res = f1_score(y_train[a], y_train_predictions, average="weighted")
            f1_test_res = f1_score(y_train[b], y_test_predictions, average="weighted")

            f1_train.append(f1_train_res)
            f1_test.append(f1_test_res)

        train_f1.append(np.mean(f1_train))
        test_f1.append(np.mean(f1_test))

        train_accuracy.append(np.mean(accuracy_list_train))
        test_accuracy.append(np.mean(accuracy_list_test))

    """
    Things we want:
        avg_train_acc_list
        avg_test_acc_list
        avg_train_f1_list
        avg_test_f1_list
    """
    percent_indicies = [i / 20 for i in range(1, 21)]

    plt.clf()

    plt.plot(percent_indicies, train_accuracy, label="Train Accuracy")
    plt.plot(percent_indicies, test_accuracy, label="Test Accuracy")
    plt.plot(percent_indicies, train_f1, label="F1 Train Score")
    plt.plot(percent_indicies, test_f1, label="F1 Test Score")

    plt.legend()

    plt.xlabel("% of Samples")
    plt.ylabel("F1 Score")

    plt.title(f"{name} for {dataset_name}")

    plt.grid()

    plt.savefig(f"../images/base_models/{name}_for_{dataset_name}")
    plt.show()


def main():
    for dataset in [
        # "titanic",
        "winequality-red"
    ]:
        df = pd.read_csv(f"../datasets/{dataset}.csv")

        if dataset == "titanic":
            predict_column = "Survived"
        else:
            predict_column = "quality"

        X = df.drop([predict_column], axis=1)
        y = df[predict_column]

        X = pd.get_dummies(X)
        X.fillna(X.mean(), inplace=True)

        all_classifiers = {
            "Decision Tree": DecisionTreeClassifier(),
            "K Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
            "Neural Network": MLPClassifier(),
            "Support Vector Machine": SVC(),
            "Ada Boost": AdaBoostClassifier()

        }
        for classifer in all_classifiers:
            test_plot(
                X=X,
                y=y,
                algo=all_classifiers[classifer],
                name=classifer,
                dataset_name=dataset)


if __name__ == "__main__":
    main()
