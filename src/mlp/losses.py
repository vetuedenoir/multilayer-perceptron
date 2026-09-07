import numpy as np


class BinaryCrossentropy:
         
    def forward(self, y, y_hat, eps=1e-15):
        m, n = y.shape
        
        y_hat = np.clip(y_hat, eps, 1 - eps)

        return -np.mean((y * np.log(y_hat) + (np.ones_like(y) - y)
                        * np.log(np.ones_like(y_hat) - y_hat)))
    
    def backward(self, y, y_hat, eps=1e-15):

        m, n = y.shape
        e = y_hat - y
        return e / (y_hat * (np.ones_like(m) - y_hat))
    
    def backward_X_Sigmoid(self, y, y_hat, eps=1e-15):
        # print(f"backward_X_Sigmoid y_hat.shape = {y_hat.shape} et y.shape = {y.shape}")
        return y_hat - y


class CategoricalCrossentropy:
    def forward(self, y, y_hat, eps=1e-15):
        m, n = y.shape
        
        y_hat = np.clip(y_hat, eps, 1 - eps)

        return -np.mean(np.sum(y * np.log(y_hat), axis=1))
    
    def backward(self, y, y_hat, eps=1e-15):
        m, n = y.shape
        e = y_hat - y
        return e / (y_hat * (np.ones_like(m) - y_hat))
    
    def backward_X_Softmax(self, y, y_hat, eps=1e-15):
        return y_hat - y