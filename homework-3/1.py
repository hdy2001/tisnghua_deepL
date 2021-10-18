# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %% [markdown]
# # Homework-3: MLP for MNIST Classification
# 
# ### In this homework, you need to
# - #### implement SGD optimizer (`./optimizer.py`)
# - #### implement forward and backward for FCLayer (`layers/fc_layer.py`)
# - #### implement forward and backward for SigmoidLayer (`layers/sigmoid_layer.py`)
# - #### implement forward and backward for ReLULayer (`layers/relu_layer.py`)
# - #### implement EuclideanLossLayer (`criterion/euclidean_loss.py`)
# - #### implement SoftmaxCrossEntropyLossLayer (`criterion/softmax_cross_entropy.py`)

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from network import Network
from solver import train, test
from plot import plot_loss_and_acc
from optimizer import SGD
from layers import FCLayer
from criterion import SoftmaxCrossEntropyLossLayer
from layers import ReLULayer

# %% [markdown]
# ## Load MNIST Dataset
# We use tensorflow tools to load dataset for convenience.

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape[0])


# %%
def decode_image(image):
    # Normalize from [0, 255.] to [0., 1.0], and then subtract by the mean value
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    image = image / 255.0
    image = image - tf.reduce_mean(image)
    return image

def decode_label(label):
    # Encode label with one-hot encoding
    return tf.one_hot(label, depth=10)


# %%
# Data Preprocessing
x_train = tf.data.Dataset.from_tensor_slices(x_train).map(decode_image)
y_train = tf.data.Dataset.from_tensor_slices(y_train).map(decode_label)
data_train = tf.data.Dataset.zip((x_train, y_train))

x_test = tf.data.Dataset.from_tensor_slices(x_test).map(decode_image)
y_test = tf.data.Dataset.from_tensor_slices(y_test).map(decode_label)
data_test = tf.data.Dataset.zip((x_test, y_test))

# %% [markdown]
# ## Set Hyerparameters
# You can modify hyerparameters by yourself.

# %%
batch_size = 100
max_epoch = 20
init_std = 0.01

learning_rate_SGD = 0.001
weight_decay = 0.1

disp_freq = 50

# %% [markdown]
# ### ~~You have finished homework3, congratulations!~~  
# 
# **Next, according to the requirements 4) of report:**
# ### **You need to construct a two-hidden-layer MLP, using any activation function and loss function.**
# 
# **Note: Please insert some new cells blow (using '+' bottom in the toolbar) refer to above codes. Do not modify the former code directly.**
# %% [markdown]
# ## 3. Construct a MLP with two hidden layers
# choose the number of hidden units by your own, using any activation function and loss function.  
# **Sigmoid Activation Function** and **ReLU Activation Function** will be used respectively again.

# %%
from criterion import SoftmaxCrossEntropyLossLayer
import importlib 




# %% [markdown]

# %% [markdown]
# ## 4. Change hyerparameters

# %%
from criterion import SoftmaxCrossEntropyLossLayer
import importlib 


criterion = SoftmaxCrossEntropyLossLayer()

sgd = SGD(learning_rate_SGD, weight_decay)


# %%
reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))


# %%
criterion = SoftmaxCrossEntropyLossLayer()

sgd = SGD(learning_rate_SGD, weight_decay)

reluMLP_0, relu_loss_0, relu_acc_0 = train(reluMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)


# %%
batch_size_1 = 100
max_epoch_1 = 20
init_std_1 = 0.01

learning_rate_SGD_1 = 0.01
weight_decay_1 = 0.1


# %%
reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))

sgd = SGD(learning_rate_SGD_1, weight_decay_1)
reluMLP_1, relu_loss_1, relu_acc_1 = train(reluMLP, criterion, sgd, data_train, max_epoch_1, batch_size_1, disp_freq)


# %%
batch_size_2 = 100
max_epoch_2 = 20
init_std_2 = 0.01

learning_rate_SGD_2 = 0.1
weight_decay_2 = 0.1


# %%
reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))

sgd = SGD(learning_rate_SGD_2, weight_decay_2)
reluMLP_2, relu_loss_2, relu_acc_2 = train(reluMLP, criterion, sgd, data_train, max_epoch_2, batch_size_2, disp_freq)


# %%
batch_size_3 = 100
max_epoch_3 = 20
init_std_3 = 0.01

learning_rate_SGD_3 = 0.001
weight_decay_3 = 0.01


# %%
reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))

sgd = SGD(learning_rate_SGD_3, weight_decay_3)
reluMLP_3, relu_loss_3, relu_acc_3 = train(reluMLP, criterion, sgd, data_train, max_epoch_3, batch_size_3, disp_freq)


# %%
batch_size_4 = 100
max_epoch_4 = 20
init_std_4 = 0.01

learning_rate_SGD_4 = 0.001
weight_decay_4 = 0.5


# %%
reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))

sgd = SGD(learning_rate_SGD_4, weight_decay_4)
reluMLP_4, relu_loss_4, relu_acc_4 = train(reluMLP, criterion, sgd, data_train, max_epoch_4, batch_size_4, disp_freq)


# %%
batch_size_7 = 100
max_epoch_7 = 5
init_std_7 = 0.01

learning_rate_SGD_7 = 0.001
weight_decay_7 = 0.1


# %%
reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))

sgd = SGD(learning_rate_SGD_7, weight_decay_7)
reluMLP_7, relu_loss_7, relu_acc_7 = train(reluMLP, criterion, sgd, data_train, max_epoch_7, batch_size_7, disp_freq)


# %%
batch_size_8 = 100
max_epoch_8 = 50
init_std_8 = 0.01

learning_rate_SGD_8 = 0.001
weight_decay_8 = 0.1


# %%
reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))

sgd = SGD(learning_rate_SGD_8, weight_decay_8)
reluMLP_8, relu_loss_8, relu_acc_8 = train(reluMLP, criterion, sgd, data_train, max_epoch_8, batch_size_8, disp_freq)


# %%
batch_size_9 = 50
max_epoch_9 = 20
init_std_9 = 0.01

learning_rate_SGD_9 = 0.001
weight_decay_9 = 0.1


# %%
reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))

sgd = SGD(learning_rate_SGD_9, weight_decay_9)
reluMLP_9, relu_loss_9, relu_acc_9 = train(reluMLP, criterion, sgd, data_train, max_epoch_9, batch_size_9, disp_freq)


# %%
batch_size_10 = 300
max_epoch_10 = 20
init_std_10 = 0.01

learning_rate_SGD_10 = 0.001
weight_decay_10 = 0.1


# %%
reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))

sgd = SGD(learning_rate_SGD_10, weight_decay_10)
reluMLP_10, relu_loss_10, relu_acc_10 = train(reluMLP, criterion, sgd, data_train, max_epoch_10, batch_size_10, disp_freq)


# %%
plot_loss_and_acc({'MLP_0': [relu_loss_0, relu_acc_0],
                                'MLP_1': [relu_loss_1, relu_acc_1], 'MLP_2':[relu_loss_2, relu_acc_2]})
plot_loss_and_acc({'MLP_0': [relu_loss_0, relu_acc_0],
                                'MLP_3': [relu_loss_3, relu_acc_3], 'MLP_4':[relu_loss_4, relu_acc_4]})
plot_loss_and_acc({'MLP_0': [relu_loss_0, relu_acc_0],
                                'MLP_7': [relu_loss_7, relu_acc_7], 'MLP_8':[relu_loss_8, relu_acc_8]})   
plot_loss_and_acc({'MLP_0': [relu_loss_0, relu_acc_0],
                                'MLP_9': [relu_loss_9, relu_acc_9], 'MLP10': [relu_loss_10, relu_acc_10]})
