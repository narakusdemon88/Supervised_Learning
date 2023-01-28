from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import f1_score
from time import perf_counter
import matplotlib.pyplot as plt

# Load the Titanic dataset
data = pd.read_csv("DT Scratches/titanic.csv")

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

# Create and fit the neural network
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
mlp.fit(X_train, y_train)

# Print the accuracy of the model on the test set
accuracy = mlp.score(X_test, y_test)
print("Accuracy:", accuracy)
