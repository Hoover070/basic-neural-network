"""
Perceptron Learning Algorithm (PLA) for linearly separable data
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Fit the model
    def fit(self, X, y):
        try:
            # Initialize weights and bias
            self.weights = np.zeros(X.shape[1])
            self.bias = 0

            for _ in range(self.epochs):
                for idx, x_i in enumerate(X):
                    linear_output = np.dot(x_i, self.weights) + self.bias
                    y_pred = self.activation_function(linear_output)
                    # Update weights and bias
                    update = self.learning_rate * (y[idx] - y_pred)
                    self.weights += update * x_i
                    self.bias += update
        except Exception as e:
            print(f'Error: {e}')

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    # Predict the output
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['SibSp'] = df['SibSp'].fillna(0)
    df['Parch'] = df['Parch'].fillna(0)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Sex'] = np.where(df['Sex'] == 'male', 0, 1)
    df['SibSp'] = np.where(df['SibSp'] > 0, 1, 0)
    df['Parch'] = np.where(df['Parch'] > 0, 1, 0)
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    return df


if __name__ == "__main__":
    # load in the data
    data_path = '../data'
    file_name = 'titanic_passengers.csv'
    random_seed = np.random.seed(42)
    out_path = os.path.join('..', '..', '..', 'visualizations', 'perceptron_visualizations')
    file_path = os.path.join(data_path, file_name)
    df = load_and_preprocess_data(file_path)

    # Split the data into features and target
    X = df[['Pclass', 'SibSp', 'Parch', 'Sex', 'Fare', 'Age']]
    y = df['Survived']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    perceptron = Perceptron()
    perceptron.fit(X_train_scaled, y_train.values)

    # Evaluate the model
    y_pred = perceptron.predict(X_test_scaled)

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'F1 Score: {f1_score(y_test, y_pred)}')

    # visualizing the distribution of target variable
    sns.countplot(x='Survived', data=df)
    print(df['Survived'].value_counts())
    plt_name = 'survived_countplot.png'
    plt.savefig(os.path.join(out_path, plt_name))
    plt.close()

    # visualizing the correlation matrix
    sns.heatmap(df.corr(), annot=True)
    plt_name = 'correlation_matrix.png'
    plt.savefig(os.path.join(out_path, plt_name))
    plt.close()
