from src.data_loader import get_data
from src.model import BayesianNN
from src.train import train
from src.evaluate import evaluate
from src.config import DATA_PATH, TARGET_COLUMN
from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from src.train import train, model_fn
import torch
def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_data(DATA_PATH, target_column=TARGET_COLUMN)

    print("Initializing model...")
    model = BayesianNN(input_dim=X_train.shape[1])
    guide = AutoNormal(model)
    optimizer = Adam({"lr": 0.01})
    loss = Trace_ELBO() 
    svi = SVI(model_fn, guide, optimizer, loss)

    print("Training model...")
    for step in range(1000):
        svi.step(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))

    print("Evaluating model...")
    evaluate(X_test, y_test, model, guide)

if __name__ == "__main__":
    main()