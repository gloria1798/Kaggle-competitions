import pandas as pd
import os

def load_data(train_file, test_file):
    train_file = os.path.join("data", train_file)
    test_file = os.path.join("data", test_file)
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    print(train_data.head())
    print(test_data.head())
    return train_data, test_data
