from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import f1_score
from time import perf_counter
import matplotlib.pyplot as plt


def main():
    for dataset in ["titanic", "winequality-red"]:
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
            # "activation",
            # "hidden_layer_number",
            "hidden_layer_nodes"
        ]

        for hyperparameter in hyperparameters:

            if hyperparameter == "activation":
                print("Testing Activation")
                f1_scores = {}

                # iterate through different activation types
                for activation in ['identity', 'logistic', 'tanh', 'relu']:
                    # create an MLPClassifier object with the current activation type
                    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation=activation)

                    # fit the model on the training data
                    mlp.fit(X_train, y_train)

                    # make predictions on the test set
                    y_pred = mlp.predict(X_test)

                    # calculate the F1 score
                    f1 = f1_score(y_test, y_pred, average="weighted")

                    # add the F1 score to the dictionary
                    f1_scores[activation] = f1

                # plot the F1 scores
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
                for hl in hidden_layers:
                    print(f"hidden layer: {hl}")
                    model = MLPClassifier(hidden_layer_sizes=(hl,), max_iter=3000, random_state=44)
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    f1_train.append(f1_score(y_train, y_train_pred, average="weighted"))
                    f1_test.append(f1_score(y_test, y_test_pred, average="weighted"))

                # Plot the F1 scores
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
                hidden_layer_nodes = [10, 20, 30, 40, 50]
                f1_train = []
                f1_test = []
                for hln in hidden_layer_nodes:
                    print(f"Hidden Layer Node: {hln}")
                    nn = MLPClassifier(hidden_layer_sizes=(hln,), max_iter=1000, random_state=44)
                    nn.fit(X_train, y_train)
                    y_train_pred = nn.predict(X_train)
                    y_test_pred = nn.predict(X_test)
                    f1_train.append(f1_score(y_train, y_train_pred))
                    f1_test.append(f1_score(y_test, y_test_pred))

                # Plot the F1 scores
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