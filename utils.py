import torch
import numpy as np


def kmeans(x, k, centroids=None, max_iter=None, epsilon=0.01, device='cpu'):
    '''
    x: data set of size (n, d) where n is the sample size.
    k: number of clusters
    centroids (optional): initial centroids
    max_iter (optional): maximum number of iterations
    epsilon (optional): error tolerance
    returns
    centroids: centroids found by k-means algorithm
    next_assigns: assignment vector
    mse: mean squared error
    it: number of iterations
    '''
    x = torch.tensor(x.to_numpy(), dtype=torch.float, device=device)
    if centroids is None:
        centroids = torch.zeros(k, x.shape[1], device=x.device)
        prev_assigns = torch.randint(0, k, (x.shape[0],), device=x.device)
        for i in range(k):
            if (prev_assigns == i).sum() > 0:
                centroids[i] = x[prev_assigns == i].mean(dim=0)

    distances = torch.cdist(centroids, x) ** 2
    prev_assigns = torch.argmin(distances, dim=0)

    it = 0
    prev_mse = distances[prev_assigns, torch.arange(x.shape[0])].mean()
    while True:
        for i in range(k):
            if (prev_assigns == i).sum() > 0:
                centroids[i] = x[prev_assigns == i].mean(dim=0)
        distances = torch.cdist(centroids, x) ** 2
        next_assigns = torch.argmin(distances, dim=0)
        if (next_assigns == prev_assigns).all():
            break
        else:
            prev_assigns = next_assigns
        it += 1
        mse = distances[next_assigns, torch.arange(x.shape[0])].mean()
        error = abs(prev_mse - mse) / prev_mse
        prev_mse = mse
        # print("iteration: %d, mse: %.3f" % (it, prev_mse.item()))

        if it == max_iter:
            break
        if error < epsilon:
            break

    return centroids, next_assigns, prev_mse, it


def create_conf_matrix(y_true, y_pred):
    x = np.array(y_true)
    labels = np.unique(x)
    labels_no = len(labels)
    conf_matr = np.zeros([labels_no, labels_no]).astype(int)

    for i in range(len(y_true)):
        conf_matr[y_true[i], y_pred[i]] += 1

    return conf_matr


def classification_score(matrix):
    num_class = len(matrix)
    true_values = 0
    precisions = []
    recalls = []

    for i in range(num_class):

        true_value = matrix[i][i]
        true_values += true_value
        pre_payda = 0
        recall_payda = 0

        for j in range(num_class):
            pre_payda += matrix[j][i]
            recall_payda += matrix[i][j]

        precision = true_value / pre_payda
        precisions.append(precision)

        recall = true_value / recall_payda
        recalls.append(recall)

    accuracy = true_values / np.sum(matrix)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_fmeasure = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)

    return accuracy, macro_precision, macro_recall, macro_fmeasure


def regression_score(y_true, y_pred):
    mse = np.mean((y_true-y_pred)**2)  # mean squared error
    mae = np.mean(abs(y_true-y_pred))  # mean absolute error
    sqrte = np.sqrt(mse)  # root mean squared error
    r2 = 1-(sum((y_true-y_pred)**2)/sum((y_true-np.mean(y_true))**2))  # r-squared error

    return mse, mae, sqrte, r2
