import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import f1_score
from time import perf_counter
import matplotlib.pyplot as plt
from DecisionTree import plot_results


def calculate_cross_val_score(X_train, y_train, dataset, kernel=None, C=None, algo=None, type=None):
    """
    :param X_train:
    :param y_train:
    :param dataset:
    :param kernel:
    :param C:
    """
    if kernel is None and C is None:
        # working with the default tree (no hyperparameter tuning)
        clf = SVC(cache_size=7_000)
    elif kernel is not None and C is None:
        # tweaking just max depth
        clf = SVC(kernel=kernel, cache_size=7_000)
    elif kernel is None and C is not None:
        # tweaking just min leaf
        clf = SVC(C=C, cache_size=7_000)
    else:
        # tweaking both
        clf = SVC(kernel=kernel, C=C, cache_size=7_000)

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
    plt.savefig(f"../images/{dataset}/{algo}/cross_validation_{type}.png")
    plt.show()


def main():
    for dataset in ["winequality-red"]:
    # for dataset in ["titanic"]:
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

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
                    clf = SVC(kernel=kernel, cache_size=7_000)

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
                    list2_label="Train Score",
                    dataset=dataset,
                    algo="svm",
                    type=hyperparameter)

                plot_results(
                    title=f"SVM Time for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=train_time,
                    list2=predict_time,
                    xlabel=hyperparameter,
                    ylabel="Time (Seconds)",
                    list1_label="Train Time",
                    list2_label="Predict Time",
                    dataset=dataset,
                    algo="svm",
                    type=hyperparameter)

            else:
                # hyperparameter is "C"
                train_time = []
                predict_time = []

                f1_test_scores = []
                f1_train_scores = []

                hyperparameter = "C"

                # for C in [i for i in range(1, 11)]:
                for C in [i for i in range(1, 201, 10)]:
                    clf = SVC(C=C, cache_size=7_000)

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
                    list2_label="Train Score",
                    dataset=dataset,
                    algo="svm",
                    type=hyperparameter)

                plot_results(
                    title=f"SVM Time for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=train_time,
                    list2=predict_time,
                    xlabel=hyperparameter,
                    ylabel="Time (Seconds)",
                    list1_label="Train Time",
                    list2_label="Predict Time",
                    dataset=dataset,
                    algo="svm",
                    type=hyperparameter)

                # test the default tree
            calculate_cross_val_score(X_train=X_train, y_train=y_train, dataset=dataset, algo="svm", type="default")

        # test the optimized tree
        if dataset == "titanic":
            # TODO: double check the right kernel
            calculate_cross_val_score(X_train=X_train, y_train=y_train, dataset=dataset, kernel="linear", C=177, algo="svm", type="optimized")
        else:
            calculate_cross_val_score(X_train=X_train, y_train=y_train, dataset=dataset, kernel="linear", C=185, algo="svm", type="optimized")


if __name__ == "__main__":
    main()
