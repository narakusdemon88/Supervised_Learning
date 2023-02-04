import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from time import perf_counter
import matplotlib.pyplot as plt


def plot_results(title, list1, list2, ylabel, xlabel, list1_label, list2_label, dataset, algo=None, type=None):
    x1, y1 = zip(*list1)
    x2, y2 = zip(*list2)

    plt.title(title)
    plt.plot(x1, y1, label=list1_label)
    plt.plot(x2, y2, label=list2_label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid()
    if algo is not None and type is not None:
        plt.savefig(f"../images/{dataset}/{algo}/cross_validation_{type}.png")
    plt.show()


def calculate_cross_val_score(X_train, y_train, dataset, max_depth=None, min_leaf=None, algo=None, type=None):
    """
    :param X_train:
    :param y_train:
    :param dataset:
    :param max_depth:
    :param min_leaf:
    """
    if max_depth is None and min_leaf is None:
        # working with the default tree (no hyperparameter tuning)
        clf = DecisionTreeClassifier()
    elif max_depth is not None and min_leaf is None:
        # tweaking just max depth
        clf = DecisionTreeClassifier(max_depth=max_depth)
    elif max_depth is None and min_leaf is not None:
        # tweaking just min leaf
        clf = DecisionTreeClassifier(min_samples_leaf=min_leaf)
    else:
        # tweaking both
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf)

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
    plt.title(f"Learning Curve: {dataset_upper} ({type})")
    plt.grid()
    if algo is not None and type is not None:
        plt.savefig(f"../images/{dataset}/{algo}/cross_validation_{type}.png")
    plt.show()

def main():
    datasets = ["titanic", "winequality-red"]
    # datasets = ["winequality-red"]

    for dataset in datasets:
        print(f"\nProcessing {dataset.upper()}")
        df = pd.read_csv(f"../datasets/{dataset}.csv")

        if dataset == "titanic":
            predict_col = "Survived"
        else:
            predict_col = "quality"

        X = df.drop([predict_col], axis=1)
        y = df[predict_col]

        X = pd.get_dummies(X)
        X.fillna(X.mean(), inplace=True)

        splits = 10
        k_folds = StratifiedKFold(n_splits=splits)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=0,
            test_size=0.2,
            shuffle=True)

        # List of hyperparameters
        hyperparameters = ["max_depth", "min_leaf"]

        for hyperparameter in hyperparameters:
            print(f"Testing {hyperparameter}")

            train_time = []
            predict_time = []

            f1_test_scores = []
            f1_train_scores = []
            for i in range(1, 21):
                if hyperparameter == "max_depth":
                    clf = DecisionTreeClassifier(max_depth=i)
                else:
                    clf = DecisionTreeClassifier(min_samples_leaf=i)

                f1_test_fold_scores = []
                f1_train_fold_scores = []

                test_fold_times = []
                train_fold_times = []

                for train_i, test_i in k_folds.split(X, y):
                    X_train = X.iloc[train_i]
                    X_test = X.iloc[test_i]

                    y_train = y.iloc[train_i]
                    y_test = y.iloc[test_i]

                    # time the fit and predict time
                    time1 = perf_counter()
                    clf.fit(X_train, y_train)
                    time2 = perf_counter()
                    y_pred_test = clf.predict(X_test)
                    time3 = perf_counter()


                t1 = perf_counter()
                clf.fit(X_train, y_train)
                t2 = perf_counter()
                y_pred_test = clf.predict(X_test)
                t3 = perf_counter()

                train_time.append((i, t2 - t1))
                predict_time.append((i, t3 - t2))

                # y_pred_test = clf.predict(X_test)
                y_pred_train = clf.predict(X_train)

                f1_test = f1_score(y_test, y_pred_test, average="weighted")
                f1_train = f1_score(y_train, y_pred_train, average="weighted")
                f1_test_scores.append((i, f1_test))
                f1_train_scores.append((i, f1_train))

                print(f"{hyperparameter}: {i}, F1_Score: {f1_test, f1_train}")

            plot_results(
                title=f"Decision Tree F1 Score for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                list1=f1_test_scores,
                list2=f1_train_scores,
                xlabel=hyperparameter,
                ylabel="F1 Score",
                list1_label="Test Score",
                list2_label="Train Score",
                dataset=dataset,
                algo="dt",
                type=hyperparameter)

            plot_results(
                title=f"Decision Tree Time for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                list1=train_time,
                list2=predict_time,
                xlabel=hyperparameter,
                ylabel="Time (Seconds)",
                list1_label="Train Time",
                list2_label="Predict Time",
                dataset=dataset,
                algo="dt",
                type=hyperparameter)

        # test the default tree
        calculate_cross_val_score(X_train=X_train, y_train=y_train, dataset=dataset, algo="dt", type="default")

        # test the optimized tree
        if dataset == "titanic":
            calculate_cross_val_score(X_train=X_train, y_train=y_train, dataset=dataset, max_depth=6, min_leaf=5, algo="dt", type="optimized")
        else:
            calculate_cross_val_score(X_train=X_train, y_train=y_train, dataset=dataset, max_depth=3, min_leaf=1, algo="dt", type="optimized")


if __name__ == "__main__":
    main()
