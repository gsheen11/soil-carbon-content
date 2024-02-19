import numpy as np
import matplotlib.pyplot as plt
import util
from tqdm import tqdm
import plotting

class FullBatchRegressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        X_clean = np.nan_to_num(X, nan=0.0)
        
        for epoch in tqdm(range(self.epochs)):
                
            predictions = np.dot(X_clean, self.weights) + self.bias
                
            dw = -(2) * (X_clean.T @ (y - predictions)) / n_samples
            db = np.mean(-(2) * (y - predictions))
                
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if epoch % 100 == 0:
                loss = np.nanmean((self.predict(X)- y) ** 2)
                print("Loss for Epoch", epoch, "is", loss)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class SGDRegressor:
    def __init__(self, learning_rate=0.01, epochs=5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in tqdm(range(self.epochs)):
            for i in range(n_samples):

                X_i = X[i, :]
                y_i = y[i]

                X_i_clean = np.nan_to_num(X_i, nan=0.0)
                
                prediction = np.dot(X_i_clean, self.weights) + self.bias
                
                dw = -(2) * (X_i_clean * (y_i - prediction))
                db = -(2) * (y_i - prediction)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            loss = np.nanmean((self.predict(X)- y) ** 2)
            print("Loss for Epoch", epoch, "is", loss)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def main():


    X, Y = util.load_training_data()
    X_test, Y_test = util.load_test_data()

    model = FullBatchRegressor(learning_rate=0.0001, epochs=10000)
    model.fit(X, Y)
    y_training_hat = model.predict(X)
    y_testing_hat = model.predict(X_test)

    print(model.bias, model.weights)
    print(np.mean(Y), np.std(Y))
    print(np.mean(Y_test), np.std(Y_test))

    print("training loss")
    loss = np.nanmean((y_training_hat - Y) ** 2)
    print(loss)

    print("test loss")
    loss = np.nanmean((y_testing_hat - Y_test) ** 2)
    print(loss)

    plotting.scatter(Y_test,y_testing_hat)
    plotting.residuals(Y_test,y_testing_hat)



if __name__ == "__main__":
    main()