""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11


class SoftmaxCrossEntropyLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = np.zeros(1, dtype='f')

    def forward(self, logit, gt):
        """
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

        ############################################################################
        # TODO: Put your code here
        # Calculate the average accuracy and loss over the minibatch, and
        # store in self.accu and self.loss respectively.
        # Only return the self.loss, self.accu will be used in solver.py.
        self.logit = logit  # (N,10)
        self.gt = gt  # (N,10)
        # softmax
        y = np.exp(logit) / np.sum(np.exp(logit), 1, keepdims=True) # (N, 10)
        
        # 记录每次输入数据的softmax值
        self.y = y

        y_result = np.argmax(y, 1)  # (N)
        gt_result = np.argmax(gt, 1)  # (N)

        self.loss = -np.sum(gt * np.log(y), 1)  # (N)
        self.acc = np.array(y_result == gt_result)  # (N)

        ############################################################################

        return self.loss

    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)
        # TODO: 我不确定是不是对的
        delta = self.y - self.gt  # (N, 10)
        return delta

    ############################################################################
