import numpy as np
import autodiff as ad



x = ad.Variable(name = "x")
w = ad.Variable(name = "w")
b = ad.Variable(name = "b")
labels = ad.Variable(name = "lables")


# TODO: 使用给定的Ops, 实现sigmoid函数
def sigmoid(x):
	rst = _
	return rst

# TODO: 使用给定的Ops, 实现逻辑回归的BCE损失函数
def bce_loss(xs, labels):
	loss = _
	return loss

p = sigmoid(ad.matmul_op(w, x))
loss = bce_loss(p, labels) 

grad_y_w, = ad.gradients(loss, [w])


num_features = 2
num_points = 200
num_iterations = 1000
learning_rate = 0.01

# The dummy dataset consists of two classes.
# The classes are modelled as a random normal variables with different means.

class_1 = np.random.normal(2, 0.1, (num_points / 2, num_features))
class_2 = np.random.normal(4, 0.1, (num_points / 2, num_features))
x_val = np.concatenate((class_1, class_2), axis = 0).T

x_val = np.concatenate((x_val, np.ones((1, num_points))), axis = 0)
w_val = np.random.normal(size = (1, num_features + 1))


labels_val = np.concatenate((np.zeros((class_1.shape[0], 1)), np.ones((class_2.shape[0], 1))), axis=0).T
executor = ad.Executor([loss, grad_y_w])

for i in xrange(100000):
	# evaluate the graph
	loss_val, grad_y_w_val =  executor.run(feed_dict={x:x_val, w:w_val, labels:labels_val})
	# TODO: update the parameters using SGD
	w_val = _
	if i % 1000 == 0:
		print loss_val