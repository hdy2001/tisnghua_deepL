""" Euclidean Loss Layer """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.accu = 0.
		self.loss = 0.

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
		# 我添加了记录输入变量的函数
		self.logit = logit
		self.gt = gt 
		logit_result = np.argmax(logit, 1) # (N)
		gt_result = np.argmax(gt, 1) # (N)
		self.loss =  1/2 * np.sum((logit - gt) * (logit - gt), 1) # (N)
		self.accu = np.array(logit_result == gt_result) # (N)
	    ############################################################################
		return self.loss

	def backward(self):
		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		grad = self.logit - self.gt # (N, 10)
		return grad
	    ############################################################################
