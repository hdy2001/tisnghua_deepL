import numpy as np


def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """
    # TODO: 还需要加入正则化lambda
    N = input.shape[0]
    b = np.ones((N, W.shape[1]))  # (N, C)
    o = np.dot(input, W) + b  # (N, C)
    y = np.exp(o) / np.sum(np.exp(o), 1, keepdims=True)  # (N, C)
    prediction = np.argmax(y, 1).reshape(N, 1)  # (N, 1)
    loss = -(1 / N) * np.sum(label * np.log(y)) + 1 / N * lamda / 2 * np.sum(
        np.dot(W, W.T))  # (int) TODO: 还可以优化

    gradient = 1 / N * np.dot(
        (y - label).T, input).T + 1 / N * lamda * W  # (N, C) * (N, D) ->(D, C)

    return loss, gradient, prediction
