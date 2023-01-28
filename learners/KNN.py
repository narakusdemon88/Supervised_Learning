from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import f1_score
import pandas as pd
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from DecisionTree import plot_results


def calculate_cross_val_score(X_train, y_train, dataset, metric=None, neighbors=None):
    """
    :param X_train:
    :param y_train:
    :param dataset:
    :param metric:
    :param neighbors:
    """
    if metric is None and neighbors is None:
        # working with the default tree (no hyperparameter tuning)
        clf = KNeighborsClassifier()
    elif metric is not None and neighbors is None:
        # tweaking just metric
        clf = KNeighborsClassifier(metric=metric)
    elif metric is None and neighbors is not None:
        # tweaking just min leaf
        clf = KNeighborsClassifier(n_neighbors=neighbors)
    else:
        # tweaking both
        clf = KNeighborsClassifier(metric=metric, n_neighbors=neighbors)

    training_sizes, training_scores, test_scores = learning_curve(
        clf,
        X_train,
        y_train,
        cv=10
    )

    # titanic --> Titanic
    dataset_upper = dataset[0].upper() + dataset[1:]

    training_sizes_percents = [(x/len(X_train))*100 for x in training_sizes]

    plt.plot(training_sizes_percents, training_scores.mean(axis=1), label="Training Score")
    plt.plot(training_sizes_percents, test_scores.mean(axis=1), label="Cross-Validation Score")
    plt.legend()
    plt.xlabel("Sample Size Percent")
    plt.ylabel("F1 Score")
    plt.title(f"Learning Curve: {dataset_upper}")
    plt.grid()
    plt.show()


def main():

    for dataset in ["titanic", "winequality-red"]:
        print(f"\nProcessing {dataset.upper()}")

        df = pd.read_csv(f"../datasets/{dataset}.csv")

        if dataset == "titanic":
            predict_col = "Survived"
        else:
            predict_col = "quality"

        X = df.drop([predict_col], axis=1)
        y = df[predict_col]

        # Preprocess the data
        X = pd.get_dummies(X)
        X.fillna(X.mean(), inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=0,
            test_size=0.2,
            shuffle=True)

        # for hyperparameter in ["distance", "k"]:
        for hyperparameter in ["k"]:
            print(f"Testing {hyperparameter}")

            train_time = []
            predict_time = []

            f1_test_scores = []
            f1_train_scores = []

            if hyperparameter == "distance":

                distance_metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]

                distance_metrics_dict = {}

                for metric in distance_metrics:
                    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)

                    # time the fit and prediction times
                    t1 = perf_counter()
                    knn.fit(X_train, y_train)
                    t2 = perf_counter()
                    y_pred_test = knn.predict(X_test)
                    t3 = perf_counter()

                    # store times to list
                    train_time.append((metric, t2 - t1))
                    predict_time.append((metric, t3 - t2))

                    # y_pred_test = clf.predict(X_test)
                    y_pred_train = knn.predict(X_train)

                    f1_test = f1_score(y_test, y_pred_test, average="weighted")
                    f1_train = f1_score(y_train, y_pred_train, average="weighted")
                    f1_test_scores.append((metric, f1_test))
                    f1_train_scores.append((metric, f1_train))

                    print(f"{hyperparameter}: {metric}, F1_Score: {f1_test, f1_train}")
                    distance_metrics_dict[metric] = (f1_test, f1_train)

                keys = list(distance_metrics_dict.keys())
                values = [value for value in distance_metrics_dict.values()]

                # Unpack the tuple values
                x1 = [val[0] for val in values]
                x2 = [val[1] for val in values]

                N = 4

                ind = np.arange(N)  # the x locations for the groups
                width = 0.35  # the width of the bars

                fig = plt.figure()
                ax = fig.add_subplot(111)
                f1_test = ax.bar(ind, x1, width)

                f1_train = ax.bar(ind + width, x2, width)

                # add some
                ax.set_ylabel('F1 Scores')
                ax.set_xlabel("Distance Metrics")
                ax.set_title("F1 Scores for KNN")
                ax.set_xticks(ind + width / 2)
                ax.set_xticklabels(keys)

                ax.legend((f1_test[0], f1_train[0]), ("F1 Test Score", "F1 Train Score"))

                plt.show()
                # https://stackoverflow.com/questions/10369681/how-to-plot-bar-graphs-with-same-x-coordinates-side-by-side-dodged

            else:
                # Iterate through k
                for k in [i for i in range(1, 21, 1)]:  # iterate through the k's
                    knn = KNeighborsClassifier(n_neighbors=k)

                    t1 = perf_counter()
                    knn.fit(X_train, y_train)
                    t2 = perf_counter()
                    y_pred_test = knn.predict(X_test)
                    t3 = perf_counter()

                    # store times to list
                    train_time.append((k, t2 - t1))
                    predict_time.append((k, t3 - t2))

                    # y_pred_test = clf.predict(X_test)
                    y_pred_train = knn.predict(X_train)

                    f1_test = f1_score(y_test, y_pred_test, average="weighted")
                    f1_train = f1_score(y_train, y_pred_train, average="weighted")
                    f1_test_scores.append((k, f1_test))
                    f1_train_scores.append((k, f1_train))

                    print(f"{hyperparameter}: {k}, F1_Score: {f1_test, f1_train}")

            plot_results(
                title=f"KNN F1 Score for {hyperparameter[0].upper() + hyperparameter[1:]}",
                list1=f1_test_scores,
                list2=f1_train_scores,
                xlabel=hyperparameter,
                ylabel="F1 Score",
                list1_label="Test Score",
                list2_label="Train Score")

            plot_results(
                title=f"KNN Time for {hyperparameter[0].upper() + hyperparameter[1:]}",
                list1=train_time,
                list2=predict_time,
                xlabel=hyperparameter,
                ylabel="Time (Seconds)",
                list1_label="Train Time",
                list2_label="Predict Time")

            # test the default tree
            calculate_cross_val_score(X_train=X_train, y_train=y_train, dataset=dataset)

            # test the optimized tree
            calculate_cross_val_score(X_train=X_train, y_train=y_train, dataset=dataset, metric="euclidean", neighbors=7)



if __name__ == "__main__":
    main()
