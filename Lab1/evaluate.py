# coding=utf-8
import numpy as np


def predict(test_images, theta):
    scores = np.dot(test_images, theta.T)
    preds = np.argmax(scores, axis=1)
    return preds


def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    acc = (
        np.sum(
            np.equal(
                y_pred,
                y.reshape(
                    -1,
                ),
            )
        )
        / y.shape[0]
    )
    return acc
