import numpy as np
import matplotlib.pyplot as plt
import util
from tqdm import tqdm
import plotting
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import KNNImputer
import K
from torch.utils.data import TensorDataset, DataLoader

class KNNImputationLayer(nn.Module):
    def __init__(self, n_neighbors=5):
        super(KNNImputationLayer, self).__init__()
        # Initialize the KNN imputer from sklearn.
        self.imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")

    def forward(self, x):
        # Check if there's any NaN in the tensor.
        if torch.isnan(x).any():
            # Ensure the tensor is on the CPU and convert it to numpy for imputation,
            # as KNNImputer does not work directly with GPU tensors.
            x_numpy = x.cpu().detach().numpy()

            # Apply KNN imputation.
            x_imputed = self.imputer.fit_transform(x_numpy)

            # Convert back to tensor. The device and dtype should match the original tensor.
            # Note: If the original tensor was on GPU, you'll need to move the imputed tensor back to the GPU.
            x = torch.tensor(x_imputed, device=x.device, dtype=x.dtype)
        return x

class NaNImputationLayer(nn.Module):
    def __init__(self, replacement_value=0.0):
        super(NaNImputationLayer, self).__init__()
        self.replacement_value = replacement_value

    def forward(self, x):
        nan_mask = torch.isnan(x)
        x = torch.where(nan_mask, torch.tensor(self.replacement_value, device=x.device), x)
        return x

class NeuralModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralModel, self).__init__()
        self.nan_imputation = NaNImputationLayer(replacement_value=0.0)
        # self.nan_imputation = KNNImputationLayer(n_neighbors=5)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.nan_imputation(x)
        out = self.relu(self.layer1(out))
        out = self.layer2(out)
        return out

def train_stochastic(model, learning_rate=0.0005, num_epochs=100):
    X, Y = util.load_training_data(as_tensor=True)
    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Forward pass
        for batch_X, batch_Y in data_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y.view(-1, 1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            outputs = model(X)
            loss = criterion(outputs, Y.view(-1, 1))
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def train(model, learning_rate=0.0005, num_epochs=100):
    X, Y = util.load_training_data(as_tensor=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, Y.view(-1, 1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def save_model(model, filename='model.pth'):
    torch.save(model.state_dict(), filename)

def load_model(model, filename='model.pth'):
    model.load_state_dict(torch.load(filename))

def eval_model(model):
    X, Y = util.load_training_data(as_tensor=True)
    X_test, Y_test = util.load_test_data(as_tensor=True)
    model.eval()
    with torch.no_grad():  # Context manager to turn off gradient computation
        y_training_hat = model(X)
        y_testing_hat = model(X_test)

        # Ensure the target tensors are the correct shape
        Y = Y.view_as(y_training_hat)
        Y_test = Y_test.view_as(y_testing_hat)

        # Calculate and print training loss
        training_loss = torch.mean((y_training_hat - Y) ** 2)
        print("Training Loss:", training_loss.item())

        # Calculate and print test loss
        test_loss = torch.mean((y_testing_hat - Y_test) ** 2)
        print("Test Loss:", test_loss.item())

        plotting.scatter(Y.numpy(),y_training_hat.numpy())
        # plotting.residuals(Y,y_training_hat)
        plotting.scatter(Y_test.numpy(),y_testing_hat.numpy())
        # plotting.residuals(Y_test,y_testing_hat)


def main():

    input_size = len(K.FEATURES_USED)
    hidden_size = 64
    model = NeuralModel(input_size=input_size, hidden_size=hidden_size)
    # train(model, learning_rate=.001, num_epochs=10000)
    train_stochastic(model, learning_rate=.001, num_epochs=300)
    save_model(model)
    # load_model(model)
    eval_model(model)



if __name__ == "__main__":
    main()