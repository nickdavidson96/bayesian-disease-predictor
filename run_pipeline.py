from src.data_loader import get_data
from src.model import BayesianNN
from src.train import train
from src.evaluate import evaluate
from src.config import DATA_PATH, TARGET_COLUMN

def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_data(DATA_PATH, target_column=TARGET_COLUMN)

    print("Initializing model...")
    model = BayesianNN(input_dim=X_train.shape[1])

    print("Training model...")
    train(X_train, y_train)

    print("Evaluating model...")
    evaluate(X_test, y_test, model)

if __name__ == "__main__":
    main()