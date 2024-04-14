import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(train_data, test_data):
    train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    test_data = test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
    train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
    test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

    train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
    test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

    X = train_data.drop(['Survived'], axis=1)
    y = train_data['Survived']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, test_data
