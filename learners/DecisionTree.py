import pandas as pd
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from time import perf_counter
import matplotlib.pyplot as plt


def plot_results(title, list_1, list_2, y_label, x_label, list_1_label, list_2_label, dataset, algo, type):
    x1, y1 = zip(*list_1)
    x2, y2 = zip(*list_2)

    plt.title(title)
    plt.plot(x1, y1, label=list_1_label)
    plt.plot(x2, y2, label=list_2_label)

    plt.legend()
    plt.xlabel(xlabel=x_label)
    plt.ylabel(ylabel=y_label)
    plt.grid()
    plt.savefig(f"../images/{dataset}/{algo}/{title}.png")
    plt.show()


def main():
    for dataset in ["titanic", "winequality-red"]:
        print(f"Processing {dataset}")
        data = pd.read_csv(f"../datasets/{dataset}.csv")

        if dataset == "titanic":
            predict_col = "Survived"
        else:
            predict_col = "quality"

        X = data.drop(columns=[predict_col])
        y = data[predict_col]

        X = pd.get_dummies(X)
        X.fillna(X.mean(), inplace=True)
        k_folds = StratifiedKFold(n_splits=5)

        for hyperparameter in ["max_depth", "min_leaf"]:
            f1_scores = []
            f1_scores_train = []
            fit_times = []
            pred_times = []

            for i in [i for i in range(1, 21, 1)]:

                if hyperparameter == "max_depth":
                    dt = DecisionTreeClassifier(max_depth=i)
                else:
                    dt = DecisionTreeClassifier(min_samples_leaf=i)

                fold_f1_scores_test = []
                fold_f1_scores_train = []
                fold_fit_times = []
                fold_pred_times = []

                k_fold_split = k_folds.split(X, y)

                for train_i, test_i in k_fold_split:

                    X_train = X.iloc[train_i]
                    X_test = X.iloc[test_i]
                    y_train = y.iloc[train_i]
                    y_test = y.iloc[test_i]

                    t1 = perf_counter()
                    dt.fit(X_train, y_train)
                    t2 = perf_counter()

                    y_pred = dt.predict(X_test)
                    t3 = perf_counter()

                    y_pred_train = dt.predict(X_train)

                    fit_time = t2 - t1
                    pred_time = t3 - t2

                    fold_f1_score_test = f1_score(y_test, y_pred, average="weighted")
                    fold_f1_score_train = f1_score(y_train, y_pred_train, average="weighted")

                    # append times to lists
                    fold_f1_scores_test.append(fold_f1_score_test)
                    fold_f1_scores_train.append(fold_f1_score_train)
                    fold_fit_times.append(fit_time)
                    fold_pred_times.append(pred_time)

                average_f1_score = sum(fold_f1_scores_test) / len(fold_f1_scores_test)
                average_f1_score_train = sum(fold_f1_scores_train) / len(fold_f1_scores_train)
                average_fit_time = sum(fold_fit_times) / len(fold_fit_times)
                average_pred_time = sum(fold_pred_times) / len(fold_pred_times)

                f1_scores.append((i, average_f1_score))
                f1_scores_train.append((i, average_f1_score_train))

                fit_times.append((i, average_fit_time))
                pred_times.append((i, average_pred_time))

                print(f"{i} F1:{average_f1_score}, Fit:{average_fit_time}, Pred:{average_pred_time}")
            print(f"f1 scores for each {hyperparameter} value: ", f1_scores)

            # PLOT F1 SCORES
            plot_results(
                title=f"Decision Tree F1 Score for {hyperparameter} on {dataset}",
                list_1=f1_scores,
                list_2=f1_scores_train,
                x_label=hyperparameter,
                y_label="F1 Score",
                list_1_label="Test Score",
                list_2_label="Train Score",
                dataset=dataset,
                algo="dt",
                type=f"{hyperparameter}_f1")
            # PLOT TIMES
            plot_results(
                title=f"Decision Tree Times for {hyperparameter} on {dataset}",
                list_1=fit_times,
                list_2=pred_times,
                x_label=hyperparameter,
                y_label="Time (Seconds)",
                list_1_label="Train Times",
                list_2_label="Predict Times",
                dataset=dataset,
                algo="dt",
                type=f"{hyperparameter}_times")

        # PLOT DEFAULT TREE
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=0,
            test_size=0.2,
            shuffle=True
        )
        for tree in ["default", "optimized"]:
            if tree == "default":
                dt = DecisionTreeClassifier()
            else:
                if dataset == "titanic":
                    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)
                else:
                    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1)

            train_sizes, train_scores, test_scores = learning_curve(
                estimator=dt,
                X=X_train,
                y=y_train,
                cv=10)

            train_scores_average = np.mean(train_scores, axis=1)
            test_scores_average = np.mean(test_scores, axis=1)

            plt.plot(train_sizes, train_scores_average, label="Training Score")
            plt.plot(train_sizes, test_scores_average, label="CV Score")
            plt.title(f"Learning Curve for {tree} on {dataset}")
            plt.xlabel("# of Samples")
            plt.ylabel("Performance Score")
            plt.legend()
            plt.grid()
            plt.savefig(f"../images/{dataset}/dt/Learning Curve for {tree} on {dataset}.png")
            plt.show()


if __name__ == "__main__":
    main()
