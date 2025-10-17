## Prepare a dataset
import pandas as pd
df = pd.read_csv("titanic.csv").dropna()

# target to 0/1
df["Survived"] = df["Survived"].map({"Yes": 1, "No": 0}).astype(int)

# drop id-like column
if "Passenger" in df.columns:
    df = df.drop(columns=["Passenger"])

# one-hot encode features (keep all dummies; tiny net can handle)
X = pd.get_dummies(df.drop(columns=["Survived"]), drop_first=False)
y = df["Survived"].values.astype(int)

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# X, y = make_blobs(n_samples=1000, n_features=2, centers=2, random_state=2023)
print(X)
print(X.shape)
print(y)
print(y.shape)
#
# # Plot the blobs to visualize the classification problem
# plt.scatter(x=X[:,0], y=X[:,1], marker=".", c=y, cmap="RdBu")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Blob distribution')
# plt.show()

# Train and test partitions
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.30, random_state=2023, stratify=y
)


# partial credit for this class (with modifications): https://realpython.com/python-ai-neural-network/, by Déborah Mesquita

import numpy as np

class TwoNodeNN:
    def __init__(self, learning_rate, n_features):
        # Hidden layer: 2 neurons
        self.W1 = np.random.randn(2, n_features) * 0.01   # (2, d)
        self.b1 = np.zeros(2)                              # (2,)
        # Output layer: 1 neuron (binary classification)
        self.W2 = np.random.randn(1, 2) * 0.01             # (1, 2)
        self.b2 = 0.0                                      # scalar
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_deriv_from_sigmoid(self, s):
        # given s = sigmoid(z)
        return s * (1.0 - s)

    # -------- forward pass --------
    def _forward_single(self, x):
        # x: (d,)
        z1 = self.W1 @ x + self.b1            # (2,)
        a1 = self._sigmoid(z1)                # (2,)
        z2 = self.W2 @ a1 + self.b2           # (1,)
        yhat = self._sigmoid(z2)[0]           # scalar
        return z1, a1, z2.item(), yhat

    # Vectorized predict_proba for grid/batches
    def predict_proba(self, X):
        """
        X: (n, d) or (d,) or pandas DataFrame
        returns probs shape (n,) or scalar
        """
        if hasattr(X, "values"):              # DataFrame/Series
            X = X.values
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            _, a1, _, yhat = self._forward_single(X)
            return yhat
        else:
            # Vectorized forward
            # Z1 = X @ W1.T + b1 -> (n,2)
            Z1 = X @ self.W1.T + self.b1
            A1 = self._sigmoid(Z1)
            Z2 = A1 @ self.W2.T + self.b2     # (n,1)
            Yhat = self._sigmoid(Z2).reshape(-1)
            return Yhat

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # -------- backprop (MSE) for one sample --------
    def _compute_gradients(self, x, y):
        # Forward
        z1, a1, z2, yhat = self._forward_single(x)

        # Loss L = (yhat - y)^2
        dL_dyhat = 2.0 * (yhat - y)                            # scalar
        dyhat_dz2 = self._sigmoid_deriv_from_sigmoid(1/(1+np.exp(-z2)))
        # More stable: use yhat directly:
        dyhat_dz2 = yhat * (1.0 - yhat)
        delta2 = dL_dyhat * dyhat_dz2                          # scalar

        # Output layer grads
        dL_dW2 = delta2 * a1.reshape(1, -1)                    # (1,2)
        dL_db2 = delta2                                        # scalar

        # Backprop to hidden
        # delta1 = (W2^T * delta2) ⊙ sigmoid'(z1)
        sigp_z1 = self._sigmoid_deriv_from_sigmoid(a1)         # (2,)
        delta1 = (self.W2.T.reshape(2,) * delta2) * sigp_z1    # (2,)

        # Hidden layer grads
        dL_dW1 = delta1.reshape(-1, 1) @ x.reshape(1, -1)      # (2,d)
        dL_db1 = delta1                                         # (2,)

        return dL_dW1, dL_db1, dL_dW2, dL_db2

    def _update(self, dW1, db1, dW2, db2):
        lr = self.learning_rate
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    # def train(self, X, y, iterations=20000, report_every=200):
    #     errs = []
    #     n = len(y)
    #     for it in range(iterations):
    #         i = np.random.randint(n)
    #         dW1, db1, dW2, db2 = self._compute_gradients(X[i], y[i])
    #         self._update(dW1, db1, dW2, db2)
    #         if it % report_every == 0:
    #             yhat_all = self.predict_proba(X)
    #             errs.append(np.mean((yhat_all - y) ** 2))
    #     return errs

    def train(self, X, y, iterations=20000, report_every=200):
        # Ensure NumPy for safe row indexing
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        errs = []
        n = len(y)
        for it in range(iterations):
            i = np.random.randint(n)
            dW1, db1, dW2, db2 = self._compute_gradients(X[i], y[i])
            self._update(dW1, db1, dW2, db2)
            if it % report_every == 0:
                yhat_all = self.predict_proba(X)
                errs.append(np.mean((yhat_all - y) ** 2))
        return errs

# Plot the results

import matplotlib.pyplot as plt
import pandas as pd

learning_rate = 0.1
neural_network = TwoNodeNN(learning_rate, n_features=X_train.shape[1])
training_error = neural_network.train(X_train, Y_train, 20000)
test_error = neural_network.train(X_test, Y_test, 20000)

#print(neural_network.predict([-4,-5]))

plt.plot(training_error)
plt.plot(test_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")
plt.show()

# Plot the decision boundary. Create a mesh of x and y points. Then
# predict the label. Then plot those with color.
# X1 = np.arange(-7, 6, 0.1)
# X2 = np.arange(-11, 11, 0.1)

# create a rectangular grid out of two given one-dimensional arrays
# X1, X2 = np.meshgrid(X1, X2)

# X_decision = pd.DataFrame({"A": np.reshape(X1,28600), "B": np.reshape(X2,28600)})
#
# Z = neural_network.predict(X_decision)
#
# plt.scatter(x=X_decision["A"],y=X_decision["B"], marker = ".", c=Z, cmap="cool")
# plt.scatter(x=X[:,0], y=X[:,1], marker = ".", c=y, cmap="RdBu")
# plt.show()
yhat_train = neural_network.predict(X_train)
yhat_test  = neural_network.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

# print("Training accuracy:", accuracy_score(Y_train, yhat_train))
# print("Training confusion matrix:\n", confusion_matrix(Y_train, yhat_train), "\n")
print("Test accuracy:", accuracy_score(Y_test, yhat_test))
print("Test confusion matrix:\n", confusion_matrix(Y_test, yhat_test))