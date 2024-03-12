import numpy as np
import matplotlib.pyplot as plt
import util
from tqdm import tqdm
import plotting
import torch
import torch.nn as nn
import torch.optim as optim

class NaNImputationLayer(nn.Module):
    def __init__(self, replacement_value=0.0):
        super(NaNImputationLayer, self).__init__()
        self.replacement_value = replacement_value

    def forward(self, x):
        # Check for NaN values in the input tensor
        nan_mask = torch.isnan(x)
        # Replace NaN values with the specified replacement value
        x = torch.where(nan_mask, torch.tensor(self.replacement_value, device=x.device), x)
        return x

class NeuralModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralModel, self).__init__()
        self.nan_imputation = NaNImputationLayer(replacement_value=0.0)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)  # Output size is 1 for regression

    def forward(self, x):
        out = self.nan_imputation(x)
        out = self.relu(self.layer1(out))
        out = self.layer2(out)
        return out


def train(model, X_train, Y_train, learning_rate=0.001, num_epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, Y_train.view(-1, 1))  # Ensure Y_train is the correct shape

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
    model.eval()  # Set the model to evaluation mode


def main():
    X, Y = util.load_training_data()
    X_test, Y_test = util.load_test_data()
    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    Y_test = torch.tensor(Y_test, dtype=torch.float)

    input_size = X.size(1)
    hidden_size = 64
    model = NeuralModel(input_size=input_size, hidden_size=hidden_size)
    train(model, X, Y, learning_rate=.01, num_epochs=10000)
    save_model(model) # and load_model()

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



if __name__ == "__main__":
    main()