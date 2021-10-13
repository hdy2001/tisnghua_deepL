""" Optimizer Class """

import numpy as np


class SGD():
    def __init__(self, learningRate, weightDecay):
        self.learningRate = learningRate
        self.weightDecay = weightDecay

    # One backpropagation step, update weights layer by layer
    def step(self, model):
        layers = model.layerList
        for layer in layers:
            if layer.trainable:
                ############################################################################
                # TODO: Put your code here
                # Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.
                # Do not forget the weightDecay term.
                N = layer.num_input
                layer.diff_W = -(layer.grad_W + self.weightDecay * layer.W /
                                 N) * self.learningRate
                layer.diff_b = -layer.grad_b * self.learningRate
                # TODO: diff_b需要加上正则化项吗
                ############################################################################

                # Weight update
                layer.W += layer.diff_W
                layer.b += layer.diff_b
