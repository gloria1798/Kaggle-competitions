from data_loader import load_data
from preprocessing import preprocess_data
from models import create_dense_model, create_cnn_model, create_rnn_model
from train_and_evaluate import train_and_evaluate_model

def main():
    # Cargar los datos
    train_data, test_data = load_data('train.csv', 'test.csv')
    print(train_data.head())
    print(test_data.head())

    # Preprocesar los datos
    X_train, X_val, y_train, y_val, test_data_processed = preprocess_data(train_data, test_data)

    # Definir diferentes modelos de deep learning
    models = [
        create_dense_model(input_shape=X_train.shape[1:]),
        create_cnn_model(input_shape=X_train.shape[1:]),
        create_rnn_model(input_shape=X_train.shape[1:])
    ]

    # Entrenar y evaluar cada modelo
    for i, model in enumerate(models):
        print(f"Entrenando Modelo {i+1}...")
        scores = train_and_evaluate_model(model, X_train, X_val, y_train, y_val)
        print(f"Modelo {i+1} - Accuracy: {scores[1]*100:.2f}%")

if __name__ == "__main__":
    main()
