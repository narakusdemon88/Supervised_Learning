import time

from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import f1_score
from time import perf_counter
import matplotlib.pyplot as plt
from DecisionTree import plot_results


def calculate_cross_val_score(X_train, y_train, dataset, kernel=None, C=None):
    """
    :param X_train:
    :param y_train:
    :param dataset:
    :param kernel:
    :param C:
    """
    if kernel is None and C is None:
        # working with the default tree (no hyperparameter tuning)
        clf = SVC()
    elif kernel is not None and C is None:
        # tweaking just max depth
        clf = SVC(kernel=kernel)
    elif kernel is None and C is not None:
        # tweaking just min leaf
        clf = SVC(C=C)
    else:
        # tweaking both
        clf = SVC(kernel=kernel, C=C)

    training_sizes, training_scores, test_scores = learning_curve(
        clf,
        X_train,
        y_train,
        cv=10
    )

    # titanic --> Titanic
    dataset_upper = dataset[0].upper() + dataset[1:]

    training_sizes_percents = [(i/len(X_train))*100 for i in training_sizes]

    plt.plot(training_sizes_percents, training_scores.mean(axis=1), label="Training Score")
    plt.plot(training_sizes_percents, test_scores.mean(axis=1), label="Cross-Validation Score")
    plt.legend()
    plt.xlabel("Sample Size %")
    plt.ylabel("F1 Score")
    plt.title(f"Learning Curve: {dataset_upper}")
    plt.grid()
    plt.show()


def main():
    # for dataset in ["titanic", "winequality-red"]:
    for dataset in ["titanic"]:
        print(f"Processing {dataset}")

        df = pd.read_csv(f"../datasets/{dataset}.csv")

        if dataset == "titanic":
            predict_col = "Survived"
        else:
            predict_col = "quality"

        X = df.drop([predict_col], axis=1)
        y = df[predict_col]

        X = pd.get_dummies(X)
        X.fillna(X.mean(), inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=0,
            test_size=0.2,
            shuffle=True)

        # hyperparameters = ["kernel", "C"]


        for hyperparameter in ["C"]:
            if hyperparameter == "kernel":
                kernels = [
                    "linear",
                    "poly",
                    "rbf",
                    "sigmoid"
                ]

                train_time = []
                predict_time = []

                f1_test_scores = []
                f1_train_scores = []

                for kernel in kernels:
                    clf = SVC(kernel=kernel)

                    t1 = time.perf_counter()
                    clf.fit(X_train, y_train)
                    t2 = time.perf_counter()
                    y_pred_test = clf.predict(X_test)
                    t3 = perf_counter()

                    train_time.append((kernel, t2 - t1))
                    predict_time.append((kernel, t3 - t2))

                    # y_pred_test = clf.predict(X_test)
                    y_pred_train = clf.predict(X_train)

                    f1_test = f1_score(y_test, y_pred_test, average="weighted")
                    f1_train = f1_score(y_train, y_pred_train, average="weighted")
                    f1_test_scores.append((kernel, f1_test))
                    f1_train_scores.append((kernel, f1_train))

                    print(f"{hyperparameter}: {kernel}, F1_Score: {f1_test, f1_train}")

                plot_results(
                    title=f"SVM F1 Score for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=f1_test_scores,
                    list2=f1_train_scores,
                    xlabel=hyperparameter,
                    ylabel="F1 Score",
                    list1_label="Test Score",
                    list2_label="Train Score")

                plot_results(
                    title=f"SVM Time for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=train_time,
                    list2=predict_time,
                    xlabel=hyperparameter,
                    ylabel="Time (Seconds)",
                    list1_label="Train Time",
                    list2_label="Predict Time")

            else:
                # hyperparameter is "C"
                train_time = []
                predict_time = []

                f1_test_scores = []
                f1_train_scores = []

                # for C in [i for i in range(1, 11)]:
                for C in [i for i in range(1, 201, 10)]:
                    clf = SVC(C=C)

                    t1 = time.perf_counter()
                    clf.fit(X_train, y_train)
                    t2 = time.perf_counter()
                    y_pred_test = clf.predict(X_test)
                    t3 = perf_counter()

                    train_time.append((C, t2 - t1))
                    predict_time.append((C, t3 - t2))

                    # y_pred_test = clf.predict(X_test)
                    y_pred_train = clf.predict(X_train)

                    f1_test = f1_score(y_test, y_pred_test, average="weighted")
                    f1_train = f1_score(y_train, y_pred_train, average="weighted")
                    f1_test_scores.append((C, f1_test))
                    f1_train_scores.append((C, f1_train))

                    print(f"{hyperparameter}: {C}, F1_Score: {f1_test, f1_train}")

                plot_results(
                    title=f"SVM F1 Score for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=f1_test_scores,
                    list2=f1_train_scores,
                    xlabel=hyperparameter,
                    ylabel="F1 Score",
                    list1_label="Test Score",
                    list2_label="Train Score")

                plot_results(
                    title=f"SVM Time for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=train_time,
                    list2=predict_time,
                    xlabel=hyperparameter,
                    ylabel="Time (Seconds)",
                    list1_label="Train Time",
                    list2_label="Predict Time")

                # test the default tree
            calculate_cross_val_score(X_train=X_train, y_train=y_train, dataset=dataset)

            # test the optimized tree
            calculate_cross_val_score(X_train=X_train, y_train=y_train, dataset=dataset, kernel=10, C=1)


if __name__ == "__main__":
    main()
