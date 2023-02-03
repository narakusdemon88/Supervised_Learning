from time import perf_counter
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import f1_score
from DecisionTree import plot_results
import numpy as np
import matplotlib.pyplot as plt


def calculate_cross_val_score(X_train, y_train, dataset, weak_learners, learning_rate, algo=None, type=None):
    """
    :param X_train:
    :param y_train:
    :param dataset:
    :param weak_learners:
    :param learning_rate:
    :return:
    """
    if weak_learners is None and learning_rate is None:
        # working with the default tree (no hyperparameter tuning)
        clf = AdaBoostClassifier()
    elif weak_learners is not None and learning_rate is None:
        # tweaking just max depth
        clf = AdaBoostClassifier(n_estimators=weak_learners)
    elif weak_learners is None and learning_rate is not None:
        # tweaking just min leaf
        clf = AdaBoostClassifier(learning_rate=learning_rate)
    else:
        # tweaking both
        clf = AdaBoostClassifier(n_estimators=weak_learners, learning_rate=learning_rate)

    training_sizes, training_scores, test_scores = learning_curve(
        clf,
        X_train,
        y_train,
        cv=10
    )

    # titanic --> Titanic
    dataset_upper = dataset[0].upper() + dataset[1:]

    training_sizes_percents = [(i / len(X_train)) * 100 for i in training_sizes]

    plt.plot(training_sizes_percents, training_scores.mean(axis=1), label="Training Score")
    plt.plot(training_sizes_percents, test_scores.mean(axis=1), label="Cross-Validation Score")
    plt.legend()
    plt.xlabel("Sample Size %")
    plt.ylabel("F1 Score")
    plt.title(f"Learning Curve: {dataset_upper}")
    plt.grid()
    if algo is not None and type is not None:
        plt.savefig(f"../images/{dataset}/{algo}/cross_validation_{type}.png")
    plt.show()


def main():
    for dataset in ["titanic", "winequality-red"]:
        print(f"Processing {dataset}")

        df = pd.read_csv(f"../datasets/{dataset}.csv")

        if dataset == "titanic":
            predict_col = "Survived"
        else:  # wine quality dataset
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

        # test hyperparamaters. n_estimators = # of weak learners
        # learning_rate = learning rate
        # base_estimator
        for hyperparameter in ["# of weak learners", "learning rate"]:
            if hyperparameter == "# of weak learners":
                # weak_learners_range = [i for i in range(1, 101, 1)]

                train_time = []
                predict_time = []

                f1_test_scores = []
                f1_train_scores = []

                for weak_learner_number in [i for i in range(1, 101, 1)]:

                    clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=weak_learner_number)
                    t1 = perf_counter()
                    clf.fit(X_train, y_train)
                    t2 = perf_counter()
                    y_pred_test = clf.predict(X_test)
                    t3 = perf_counter()

                    train_time.append((weak_learner_number, t2 - t1))
                    predict_time.append((weak_learner_number, t3 - t2))

                    # y_pred_test = clf.predict(X_test)
                    y_pred_train = clf.predict(X_train)

                    f1_test = f1_score(y_test, y_pred_test, average="weighted")
                    f1_train = f1_score(y_train, y_pred_train, average="weighted")
                    f1_test_scores.append((weak_learner_number, f1_test))
                    f1_train_scores.append((weak_learner_number, f1_train))

                    print(f"{hyperparameter}: {weak_learner_number}, F1_Score: {f1_test, f1_train}")

                plot_results(
                    title=f"SVM F1 Score for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=f1_test_scores,
                    list2=f1_train_scores,
                    xlabel=hyperparameter,
                    ylabel="F1 Score",
                    list1_label="Test Score",
                    list2_label="Train Score",
                    dataset=dataset,
                    algo="boost",
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
                    algo="boost",
                    type=hyperparameter)

            else:
                # hyperparameter == "learning rate"
                train_time = []
                predict_time = []

                f1_test_scores = []
                f1_train_scores = []

                for rate in np.linspace(0.000001, 1, 100):
                    clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(), learning_rate=rate)
                    t1 = perf_counter()
                    clf.fit(X_train, y_train)
                    t2 = perf_counter()
                    y_pred_test = clf.predict(X_test)
                    t3 = perf_counter()

                    train_time.append((rate, t2 - t1))
                    predict_time.append((rate, t3 - t2))

                    # y_pred_test = clf.predict(X_test)
                    y_pred_train = clf.predict(X_train)

                    f1_test = f1_score(y_test, y_pred_test, average="weighted")
                    f1_train = f1_score(y_train, y_pred_train, average="weighted")
                    f1_test_scores.append((rate, f1_test))
                    f1_train_scores.append((rate, f1_train))

                    print(f"{hyperparameter}: {rate}, F1_Score: {f1_test, f1_train}")

                plot_results(
                    title=f"SVM F1 Score for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=f1_test_scores,
                    list2=f1_train_scores,
                    xlabel=hyperparameter,
                    ylabel="F1 Score",
                    list1_label="Test Score",
                    list2_label="Train Score",
                    dataset=dataset,
                    algo="boost",
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
                    algo="boost",
                    type=hyperparameter)

        # test the default tree
        calculate_cross_val_score(
            X_train=X_train, y_train=y_train, dataset=dataset, weak_learners=46, learning_rate=0.01, algo="boost", type="default")


        # test the optimized tree
        if dataset == "titanic":
            calculate_cross_val_score(
                X_train=X_train, y_train=y_train, dataset=dataset, weak_learners=1, learning_rate=0.000001, algo="boost", type="optimized")
        else:
            calculate_cross_val_score(
                X_train=X_train, y_train=y_train, dataset=dataset, weak_learners=46, learning_rate=0.01, algo="boost", type="optimized")


if __name__ == "__main__":
    main()
