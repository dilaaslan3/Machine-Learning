import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:
    def __init__(self, X_train, y_train, k):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, knn_type):
        y_pred = [self._predict(x_test, knn_type) for x_test in X_test]
        return np.array(y_pred)

    def _predict(self, x_test, knn_type):
        distances = [euclidean_distance(x_test, x_train) for x_train in self.X_train]

        # Sort by distance and return indices of the first k neighbors
        k_index = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_index]

        if knn_type == "clf":
            # return the most common class label
            most_k = Counter(k_neighbor_labels).most_common(1)
            return most_k[0][0]

        elif knn_type == "reg":
            # return the avarage of k labels
            return np.mean(k_neighbor_labels)
