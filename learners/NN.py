from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import f1_score
from time import perf_counter
import matplotlib.pyplot as plt


def plot_loss_curve(data):
    # Split the data into features and target
    X = data.drop("Survived", axis=1)
    y = data["Survived"]

    # Pre-process the data
    X = pd.get_dummies(X)
    X.fillna(X.mean(), inplace=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scale the data to improve the performance of the neural network
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create an instance of the MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=300, early_stopping=True)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Get the loss history
    loss_history = model.loss_curve_

    # Plot the loss curve
    plt.plot(loss_history)
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()


def main():
    for dataset in ["titanic", "winequality-red"]:
        print(f"\nProcessing {dataset.upper()}")

        df = pd.read_csv(f"../datasets/{dataset}.csv")

        if dataset == "titanic":
            predict_col = "Survived"
        else:
            predict_col = "quality"

        # Plot the loss curve (see function above)
        # plot_loss_curve(data=df)

        # Split the data into features and target
        X = df.drop([predict_col], axis=1)
        y = df[predict_col]

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
            "hidden_layers",
            ""
        ]

        for hyperparameter in hyperparameters:

            if hyperparameter == "hidden_layers":
                print("Testing Hidden Layers")
                f1_scores = []

                # Iterate through different numbers of hidden layers
                for num_layers in range(1, 11):
                    # Create an instance of the MLPClassifier
                    model = MLPClassifier(hidden_layer_sizes=(10,) * num_layers, max_iter=300, early_stopping=True)

                    # Fit the model to the training data
                    model.fit(X_train, y_train)

                    # Predict on the test data
                    y_pred = model.predict(X_test)

                    # Get the F1 score
                    f1 = f1_score(y_test, y_pred)

                    # Append the F1 score to the list
                    f1_scores.append(f1)

                # Plot the F1 scores
                plt.plot(range(1, 11), f1_scores)
                plt.xlabel('Number of hidden layers')
                plt.ylabel('F1 score')
                plt.title('F1 score vs Number of hidden layers')
                plt.show()

            # # Create and fit the neural network
            # mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
            # mlp.fit(X_train, y_train)
            #
            # # time the fit and prediction times
            # t1 = perf_counter()
            # mlp.fit(X_train, y_train)
            # t2 = perf_counter()
            # y_pred_test = mlp.predict(X_test)
            # t3 = perf_counter()
            #
            # # Print the accuracy of the model on the test set
            # accuracy = mlp.score(X_test, y_test)
            # f1 = f1_score(y_test, y_pred_test, average='weighted')
            # print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()