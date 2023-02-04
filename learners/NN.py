from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import f1_score
from time import perf_counter
import matplotlib.pyplot as plt
from DecisionTree import plot_results


def calculate_cross_val_score(X_train, y_train, dataset, activation=None, hidden_layer_sizes=None, algo=None, type=None):
    """
    :param X_train:
    :param y_train:
    :param dataset:
    :param activation:
    :param hidden_layer_sizes:
    """
    if activation is None and hidden_layer_sizes is None:
        # working with the default tree (no hyperparameter tuning)
        clf = MLPClassifier()
    elif activation is not None and hidden_layer_sizes is None:
        # tweaking just activation
        clf = MLPClassifier(activation=activation)
    elif activation is None and hidden_layer_sizes is not None:
        # tweaking just min leaf
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    else:
        # tweaking both
        clf = MLPClassifier(activation=activation, hidden_layer_sizes=hidden_layer_sizes)

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
    plt.title(f"Learning Curve: {dataset_upper} ({type})")
    plt.grid()
    if algo is not None and type is not None:
        plt.savefig(f"../images/{dataset}/{algo}/cross_validation_{type}.png")
    plt.show()


def main():
    for dataset in ["winequality-red"]:
        print(f"\nProcessing {dataset.upper()}")

        data = pd.read_csv(f"../datasets/{dataset}.csv")

        if dataset == "titanic":
            predict_col = "Survived"
        else:
            predict_col = "quality"

        X = data.drop(predict_col, axis=1)
        y = data[predict_col]

        X = pd.get_dummies(X)
        X.fillna(X.mean(), inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        hyperparameters = [
            "activation",
            "hidden_layer_number",
            "hidden_layer_nodes"
        ]

        for hyperparameter in hyperparameters:

            if hyperparameter == "activation":
                print("Testing Activation")
                f1_scores = {}

                train_time = []
                predict_time = []

                # iterate through different activation types
                for activation in ['identity', 'logistic', 'tanh', 'relu']:
                    # create an MLPClassifier object with the current activation type
                    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation=activation)

                    t1 = perf_counter()
                    mlp.fit(X_train, y_train)
                    t2 = perf_counter()
                    y_pred = mlp.predict(X_test)
                    t3 = perf_counter()

                    train_time.append((activation, t2 - t1))
                    predict_time.append((activation, t3 - t2))

                    f1 = f1_score(y_test, y_pred, average="weighted")

                    f1_scores[activation] = f1

                plot_results(
                    title=f"Neural Network Time for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=train_time,
                    list2=predict_time,
                    xlabel=hyperparameter,
                    ylabel="Time (Seconds",
                    list1_label="Train Time",
                    list2_label="Predict Time",
                    dataset=dataset,
                    algo="nn",
                    type=hyperparameter
                )

                plt.bar(f1_scores.keys(), f1_scores.values())
                plt.title(f"{dataset} F1 Score")
                plt.xlabel("Activation Type")
                plt.ylabel("F1 Score")
                plt.show()

                ######### End of activation type comparison

            elif hyperparameter == "hidden_layer_number":
                # Iterate through different numbers of hidden layers
                hidden_layers = [i for i in range(1, 11, 1)]
                f1_train = []
                f1_test = []

                train_time = []
                predict_time = []

                for hl in hidden_layers:
                    print(f"hidden layer: {hl}")
                    model = MLPClassifier(hidden_layer_sizes=(hl,), max_iter=3000)
                    t1 = perf_counter()
                    model.fit(X_train, y_train)
                    t2 = perf_counter()
                    y_test_pred = model.predict(X_test)
                    t3 = perf_counter()
                    y_train_pred = model.predict(X_train)

                    train_time.append((hl, t2 - t1))
                    predict_time.append((hl, t3 - t2))

                    f1_train.append(f1_score(y_train, y_train_pred, average="weighted"))
                    f1_test.append(f1_score(y_test, y_test_pred, average="weighted"))

                # plot_results(
                #     title=f"Neural Network F1 Score for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                #     list1=f1_test,
                #     list2=f1_train,
                #     xlabel=hyperparameter,
                #     ylabel="F1 Score",
                #     list1_label="Test Score",
                #     list2_label="Train Score",
                #     dataset=dataset,
                #     algo="nn",
                #     type=hyperparameter)

                plot_results(
                    title=f"Neural Network Time for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=train_time,
                    list2=predict_time,
                    xlabel=hyperparameter,
                    ylabel="Time (Seconds)",
                    list1_label="Train Time",
                    list2_label="Predict Time",
                    dataset=dataset,
                    algo="nn",
                    type=hyperparameter)

                plt.plot(hidden_layers, f1_train, label='F1 Train Score')
                plt.plot(hidden_layers, f1_test, label='F1 Test Score')
                plt.title(f"Neural Network F1 Score for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}")
                plt.xlabel("# of Hidden Layers")
                plt.ylabel("F1 Score")
                plt.grid()
                plt.legend()
                plt.savefig(f"../images/{dataset}/nn/f1_hidden_layers.png")
                plt.show()

            else:  # hidden layer nodes
                # Iterate through different numbers of hidden layer nodes
                hidden_layer_nodes = [i for i in range(10, 101, 10)]

                train_time = []
                predict_time = []

                f1_train = []
                f1_test = []
                for hln in hidden_layer_nodes:
                    print(f"Hidden Layer Node: {hln}")
                    nn = MLPClassifier(hidden_layer_sizes=(hln,), max_iter=1000, random_state=44)
                    t1 = perf_counter()
                    nn.fit(X_train, y_train)
                    t2 = perf_counter()
                    y_test_pred = nn.predict(X_test)
                    t3 = perf_counter()

                    train_time.append((hln, t2 - t1))
                    predict_time.append((hln, t3 - t2))

                    y_train_pred = nn.predict(X_train)

                    f1_train.append(f1_score(y_train, y_train_pred, average="weighted"))
                    f1_test.append(f1_score(y_test, y_test_pred, average="weighted"))

                # plot_results(
                #     title=f"Neural Network F1 Score for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                #     list1=f1_test,
                #     list2=f1_train,
                #     xlabel=hyperparameter,
                #     ylabel="F1 Score",
                #     list1_label="Test Score",
                #     list2_label="Train Score",
                #     dataset=dataset,
                #     algo="nn",
                #     type=hyperparameter)

                plot_results(
                    title=f"Neural Network Time for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=train_time,
                    list2=predict_time,
                    xlabel=hyperparameter,
                    ylabel="Time (Seconds)",
                    list1_label="Train Time",
                    list2_label="Predict Time",
                    dataset=dataset,
                    algo="nn",
                    type=hyperparameter)

                plt.plot(hidden_layer_nodes, f1_train, label='F1 Train Score')
                plt.plot(hidden_layer_nodes, f1_test, label='F1 Test Score')
                plt.xlabel("# of Hidden Layers Nodes")
                plt.ylabel('F1 score')
                plt.title(f"Neural Network F1 Score for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}")
                plt.grid()
                plt.legend()
                plt.savefig(f"../images/{dataset}/nn/f1_hidden_layers.png")
                plt.show()

        # test the default tree
        calculate_cross_val_score(X_train=X_train, y_train=y_train, dataset=dataset, algo="nn", type="default")

        if dataset == "titanic":
            calculate_cross_val_score(
                X_train=X_train, y_train=y_train, dataset=dataset, activation="identity", hidden_layer_sizes=(10, 50, 20), algo="nn", type="optimized")
        else:
            calculate_cross_val_score(
                X_train=X_train, y_train=y_train, dataset=dataset, activation="logistic", hidden_layer_sizes=(50, 100, 20), algo="nn", type="optimized")


if __name__ == "__main__":
    main()