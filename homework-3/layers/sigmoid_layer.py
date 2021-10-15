""" Sigmoid Layer """

import numpy as np
"""input = output"""


class SigmoidLayer():
    def __init__(self):
        """
		Applies the element-wise function: f(x) = 1/(1+exp(-x))
		"""
        self.trainable = False

    def forward(self, Input):
        ############################################################################
        # TODO: Put your code here
        # Apply Sigmoid activation function to Input, and return results.
        self.input = Input  # (N, input)
        return 1 / (1 + np.exp(-Input))  # (N, output)

    ############################################################################

    def backward(self, delta):
        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient using the later layer's gradient: delta
        # delta: (N, ouput)
        # 导数：f(z)(1-f(z))
        f_ = (1 / (1 + np.exp(-self.input))) * (1 - (1 / (1 + np.exp(-self.input))))  # (N, output)
        return f_ * delta  # (N, output)

    ############################################################################
