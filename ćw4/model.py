import pickle
import pprint
import cvxopt.solvers
import numpy as np
import numpy.linalg as la


from sklearn.svm import SVC

"""Implement your model, training code and other utilities here. Please note, you can generate multiple
pickled data files and merge them into a single data list."""


'''
1. DOWN     = 2
2. UP       = 0
3. RIGHT    = 1
4. LEFT     = 3
'''


class Kernel():
    def linear():
        return lambda x, y: np.inner(x, y)

    def gaussian(sigma=5):
        return lambda x, y: \
            np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))

    def radial_basis(gamma=10):
        return lambda x, y: np.exp(-gamma*la.norm(np.subtract(x, y)))


class SVM_train_model():
    def __init__(self, kernel, c):
        self._kernel = kernel
        self._c = c

    def train(self, X, y):
        lagrange_multipliers = self._compute_multipliers(X, y)
        return self._construct_predictor(X, y, lagrange_multipliers)

    def _compute_multipliers(self, X, y):
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = \
            lagrange_multipliers > 1e-5

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        bias = np.mean(
            [y_k - SVM_predicting_model(
                kernel=self._kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        return SVM_predicting_model(
            kernel=self._kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)


class SVM_predicting_model(object):
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        assert len(weights) == len(support_vector_labels)

    def predict(self, x):
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()


def import_data(file_path):
    X = []
    y = []
    with open(file_path, 'rb') as f:
        data_file = pickle.load(f)
    # pprint.pprint(data_file["block_size"])
    # pprint.pprint(data_file["bounds"])
    # pprint.pprint(data_file["data"][3])
    x_bound = data_file["bounds"][0]
    y_bound = data_file["bounds"][1]
    block_size = data_file["block_size"]
    for sample in data_file["data"]:
        game_state, y_sample = sample
        X.append(game_state_to_data_sample(game_state, x_bound, y_bound, block_size))
        y.append(y_sample.value)
    X = np.array(X)
    return X, y


def split_data(X, y):
    X_test = []
    X_train = []
    y_test = []
    y_train = []
    for i in range(len(X)):
        X_test.append(X[i]) if i % 5 == 0 else X_train.append(X[i])
        y_test.append(y[i]) if i % 5 == 0 else y_train.append(y[i])
    return X_test, X_train, y_test, y_train


def game_state_to_data_sample(game_state: dict, x_bound, y_bound, block_size):
    data = game_state
    food = data["food"]
    snake_body = data["snake_body"]
    attributes = []

    # atrybuty zwiazane z przeszkodami
    for dx, dy in [(0, block_size), (0, -block_size), (block_size, 0), (-block_size, 0)]:
        neighbor_x, neighbor_y = snake_body[-1][0] + dx, snake_body[-1][1] + dy
        obstacle_in_neighbor = (neighbor_x, neighbor_y) in snake_body or neighbor_x < 0 or neighbor_x >= x_bound or neighbor_y < 0 or neighbor_y >= y_bound
        attributes.append(int(obstacle_in_neighbor))

    # atrybuty zwiazane z jedzeniem
    for i in range(2):
        food_in_direction = int(food[i] > snake_body[-1][i])
        attributes.append(int(food_in_direction))
        food_in_direction = int(food[i] < snake_body[-1][i])
        attributes.append(int(food_in_direction))

    return attributes


if __name__ == "__main__":
    """ Example of how to read a pickled file, feel free to remove this"""
    # with open(f"data/2023-12-06_170545.pickle", 'rb') as f:
    #     data_file = pickle.load(f)
    # # pprint.pprint(data_file["block_size"])
    # # pprint.pprint(data_file["bounds"])
    # pprint.pprint(data_file["data"][3])
    # game_state = data_file["data"][3]
    # print(game_state_to_data_sample(game_state))
    file_path = "data/2023-12-06_173416.pickle"
    X, y = import_data(file_path)
    X_test, X_train, y_test, y_train = split_data(X, y)
    kernel = Kernel.radial_basis()
    c = 1e-2
    svm = SVM_train_model(kernel, c)
    svm.train(np.array(X_train), np.array(y_train))
    y_pred = svm.predict(X_test)
    correct = 0
    for i in range(len(y_pred)):
        print(f"{i}. y_pred={y_pred[i]}, y_test={y_test[i]}")
        if y_pred[i] == y_test[i]:
            correct += 1
    print(f"{correct}/{len(y_pred)} = {correct*100/len(y_pred)}%")
