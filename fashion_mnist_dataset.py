
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.datasets import mnist, fashion_mnist

(X_train_fashion, y_train_fashion), (X_test_fashion, y_test_fashion) = fashion_mnist.load_data()

X_train_fashion = X_train_fashion.reshape(len(X_train_fashion), -1) / 255.0
X_test_fashion = X_test_fashion.reshape(len(X_test_fashion), -1) / 255.0

X_train_fashion.shape,X_test_fashion.shape

class perceptron:
    def __init__(self,learning_rate=0.1,n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights=None
        self.bias=None
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for temp in range(self.n_iters):
            print(temp)
            for idx,x_i in enumerate(X):
                linear_output = np.dot(x_i,self.weights)+self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.lr + (y[idx]-y_predicted)
                self.weights += update*x_i
                self.bias += update
    def predict(self,X):
        linear_output = np.dot(X,self.weights)+self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
    def _unit_step_func(self,x):
        return np.where(x>=0,1,0)

p = perceptron(learning_rate=0.01,n_iters=100)
p.fit(X_train_fashion, y_train_fashion)

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Reduce 784 → 2
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train_fashion)

# Train your perceptron on 2D projected data
p.fit(X_train_2D, y_train_fashion)

# Meshgrid for visualization
xx, yy = np.meshgrid(
    np.linspace(X_train_2D[:, 0].min() - 1, X_train_2D[:, 0].max() + 1, 100),
    np.linspace(X_train_2D[:, 1].min() - 1, X_train_2D[:, 1].max() + 1, 100)
)

Z = p.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=y_train_fashion, cmap=plt.cm.Paired, s=10)
plt.title("Decision Boundary after PCA (2D Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

x = np.linspace(-5,5,100)
sigmoid = 1/(1+np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)

plt.figure(figsize=(8,4))
plt.plot(x, sigmoid, label='Sigmoid')
plt.plot(x, tanh, label='Tanh')
plt.plot(x, relu, label='ReLU')
plt.legend()
plt.title("Activation Functions")
plt.grid(True)
plt.show()