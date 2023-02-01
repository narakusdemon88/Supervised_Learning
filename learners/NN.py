from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import f1_score
from time import perf_counter
import matplotlib.pyplot as plt
from DecisionTree import plot_results

def main():
    for dataset in ["winequality-red"]:
        print(f"\nProcessing {dataset.upper()}")

        data = pd.read_csv(f"../datasets/{dataset}.csv")

        if dataset == "titanic":
            predict_col = "Survived"
        else:
            predict_col = "quality"

        # Split the data into features and target
        X = data.drop(predict_col, axis=1)
        y = data[predict_col]

        # Pre-process the data
        X = pd.get_dummies(X)
        X.fillna(X.mean(), inplace=True)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Scale the data to improve the performance of the neural network
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
                    list2_label="Predict Time"
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

                plot_results(
                    title=f"Neural Network Time for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=train_time,
                    list2=predict_time,
                    xlabel=hyperparameter,
                    ylabel="Time (Seconds)",
                    list1_label="Train Time",
                    list2_label="Predict Time"
                )

                plt.plot(hidden_layers, f1_train, label='F1 Train Score')
                plt.plot(hidden_layers, f1_test, label='F1 Test Score')
                plt.title(f"{dataset} Hidden Layer F1 Scores")
                plt.xlabel("# of Hidden Layers")
                plt.ylabel("F1 Score")
                plt.grid()
                plt.legend()
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

                plot_results(
                    title=f"Neural Network Time for {hyperparameter[0].upper() + hyperparameter[1:]} on {dataset}",
                    list1=train_time,
                    list2=predict_time,
                    xlabel=hyperparameter,
                    ylabel="Time (Seconds)",
                    list1_label="Train Time",
                    list2_label="Predict Time"
                )

                plt.plot(hidden_layer_nodes, f1_train, label='F1 Train Score')
                plt.plot(hidden_layer_nodes, f1_test, label='F1 Test Score')
                plt.xlabel("# of Hidden Layers Nodes")
                plt.ylabel('F1 score')
                plt.title(f"{dataset} Hidden Layer Nodes F1 Scores")
                plt.grid()
                plt.legend()
                plt.show()


if __name__ == "__main__":
    main()