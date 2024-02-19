import numpy as np
import matplotlib.pyplot as plt
import util
from tqdm import tqdm

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
        
        # Stochastic Gradient Descent
        for _ in tqdm(range(self.epochs)):
            for i in range(n_features):
                random_index = np.random.randint(0, n_samples)
                X_i = X[random_index, :]
                y_i = y[random_index]

                if not y_i or not X_i[0] or not X_i[1]:
                    continue
                
                # Calculate predictions
                prediction = np.dot(X_i, self.weights) + self.bias
                
                # Compute gradients
                dw = -(2) * (X_i * (y_i - prediction))
                db = -(2) * (y_i - prediction)
                
                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def main():


    X, Y = util.load_training_data()
    X_test, Y_test = util.load_test_data()

    model = SGDRegressor(learning_rate=0.001, epochs=5)
    model.fit(X, Y)
    y_training_hat = model.predict(X)
    y_testing_hat = model.predict(X_test)

    # print(Y[0:10])

    print(model.bias, model.weights)

    print("training loss")
    loss = np.mean((y_training_hat - Y) ** 2)
    print(loss)

    print("test loss")
    loss = np.mean((y_testing_hat - Y_test) ** 2)
    print(loss)



if __name__ == "__main__":
    main()