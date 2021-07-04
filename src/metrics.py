import numpy as np
from scipy.special import softmax


def accuracy(y_true, y_pred):
    pred_labels = y_pred.argmax(axis=1)
    true_labels = y_true.argmax(axis=1)
    hits = true_labels == pred_labels
    assert len(hits.shape) == 1
    return hits.mean()


def crossentropy(y_true, y_pred, eps=1e-9):
    y_pred = softmax(y_pred, axis=1)
    true_labels = y_true.argmax(axis=1)

    mask = np.eye(y_pred.shape[1])[true_labels]
    p = np.sum(y_pred * mask, axis=1)
    return -np.log(np.clip(p, eps, 1 - eps)).mean()
