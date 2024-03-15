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
from sklearn.model_selection import KFold

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
    
class GaussianImputationLayer(nn.Module):
    def __init__(self, data):
        super(GaussianImputationLayer, self).__init__()
        data_numpy = data.numpy()
        np_mean = np.nanmean(data_numpy, axis=0)
        np_std = np.nanstd(data_numpy, axis=0)
        self.replacement_mean = torch.tensor(np_mean, dtype=torch.float)
        self.replacement_std = torch.tensor(np_std, dtype=torch.float)

    def forward(self, x):
        nan_mask = torch.isnan(x)
        size_batch = x.shape[0]
        if self.training:
            new_samples = torch.normal(self.replacement_mean.expand(size_batch, -1),
                                       self.replacement_std.expand(size_batch, -1))
        else:
            new_samples = self.replacement_mean.expand(size_batch, -1)
        x = torch.where(nan_mask, new_samples, x)
        return x

class NaNImputationLayer(nn.Module):
    def __init__(self, replacement_value=0.0):
        super(NaNImputationLayer, self).__init__()
        self.replacement_value = replacement_value

    def forward(self, x):
        nan_mask = torch.isnan(x)
        x = torch.where(nan_mask, torch.tensor(self.replacement_value, device=x.device), x)
        return x
    
class RandomNaNDropout(nn.Module):
    def __init__(self, drop_prob=0.1):
        super(RandomNaNDropout, self).__init__()
        if not (0 <= drop_prob <= 1):
            raise ValueError("drop_prob must be in the interval [0, 1]")
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.shape) < self.drop_prob
            return torch.where(mask, torch.tensor(float('nan'), device=x.device), x)
        return x
    
class NormalizationLayer(nn.Module):
    def __init__(self, X):
        super(NormalizationLayer, self).__init__()
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0)

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-6)
    
class MaskingLayer(nn.Module):
    def __init__(self):
        super(MaskingLayer, self).__init__()

    def forward(self, x):
        # Creating a mask where True (or 1) indicates a NaN value
        mask = torch.isnan(x)
        # You might want to convert the boolean mask to float if you plan to use it in calculations
        return mask.bool(), x

class NeuralModel(nn.Module):
    def __init__(self, input_size, hidden_size, drop_prob, X):
        super(NeuralModel, self).__init__()
        # self.nan_imputation = NaNImputationLayer(replacement_value=0.0)
        # self.nan_imputation = KNNImputationLayer(n_neighbors=5)
        self.mask_layer = MaskingLayer()
        self.normalization_layer = NormalizationLayer(X)
        self.rand_dropout = RandomNaNDropout(drop_prob=drop_prob)
        self.nan_imputation = GaussianImputationLayer(X)
        self.layer1 = nn.Linear(input_size + input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.rand_dropout(x)
        mask, x = self.mask_layer(x)
        x = self.nan_imputation(x)
        x = self.normalization_layer(x)
        x = torch.cat((mask, x), dim=1)
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def train_stochastic(model, X, Y, learning_rate=0.0005, num_epochs=100):
    # X, Y = util.load_training_data(as_tensor=True)
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

def train(model, X, Y, learning_rate=0.0005, num_epochs=100):
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

def save_model(model, filename='models/model.pth'):
    torch.save(model.state_dict(), filename)

def load_model(model, filename='models/model.pth'):
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

def eval_model_rand_subset(model):
    model.eval()
    with torch.no_grad():  # Context manager to turn off gradient computation
        for i in [0,5,10,15,20,25,30,35]:
            X_test, Y_test = util.load_test_data(as_tensor=True, path="data/test_set_" + str(i) + ".csv")
            y_testing_hat = model(X_test)

            # Ensure the target tensors are the correct shape
            Y_test = Y_test.view_as(y_testing_hat)

            # Calculate and print test loss
            test_loss = torch.mean((y_testing_hat - Y_test) ** 2)
            print("Test Loss for " + str(i) + " :", test_loss.item())

            # plotting.residuals(Y,y_training_hat)
            # plotting.scatter(Y_test.numpy(),y_testing_hat.numpy())
            # plotting.residuals(Y_test,y_testing_hat)

def crossValidate(n_splits=5):
    X, Y = util.load_training_data(as_tensor=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        
        model = NeuralModel(input_size=len(K.FEATURES_USED), hidden_size=64, drop_prob=0.3, X=X)
        train_stochastic(model, X_train, Y_train, learning_rate=0.001, num_epochs=100)
        
        score =  torch.mean((model(X_val) - Y_val) ** 2)
        print("SCORE: ", score.item())
        
        scores.append(score.item())
    
    average_score = sum(scores) / n_splits
    print(f"Average score across all folds: {average_score}")

def main():

    # X, Y = util.load_training_data(as_tensor=True)
    # model = NeuralModel(input_size=len(K.FEATURES_USED), hidden_size=64, drop_prob=0.3, X=X)
    # train(model, X, Y, learning_rate=.01, num_epochs=1000)
    # train_stochastic(model, X, Y, learning_rate=.001, num_epochs=100)
    # save_model(model)
    # load_model(model)
    # eval_model(model)
    # eval_model_rand_subset(model)
    crossValidate(n_splits=3)



if __name__ == "__main__":
    main()