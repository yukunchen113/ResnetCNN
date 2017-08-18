'''
Further improvements:
	- change first shortcut to linear projection
	- make validation set, to determine when to stop training
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 
import os
from six.moves import xrange
import math 
#import cifar10_input as inp
import caltech256_input as inp
Sets = inp.Sets
batch_size= 128#the size of the minibatch
data_dir = inp.data_dir

variable_float16 = False
EXPONETIAL_MOVING_AVERAGE_DECAY = 0.999#for mean and var, total loss
N_CHANNELS = 3 # rgb
RESNET_LAYER_INFO = [1,3,4,6,2]#numbers of resnet blocks in each layer
N_CLASSES = inp.NUM_CLASSES#number of classes
N_FC = 1000
N_SML = N_CLASSES #number of neurons in fully connected layer
OUTPUT_NOISE = 0.00000#for label smoothing
START_DEPTH = 16#start depth for resnet layers
STEP_SIZE = 0.00005#initial learning rate for adam
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_DELTA = 0.00001 #small value for numerical stabilization
FIRST_CONV_LAYER_KERNEL_LENGTH = 3
FIRST_CONV_LAYER_DEPTH = 16
N_EXAMPLES_TRAIN_EPOCH = inp.N_EXAMPLES_TRAIN_EPOCH
N_EXAMPLES_EVAL_EPOCH = inp.N_EXAMPLES_EVAL_EPOCH

def variable(name, shape, initializer):
	'''
	gets variable of name, name (creates variable if DNE)
	Args:
		name: name of the variable, string
		shape: shape of the variable, list
		initializer: initial value for variable if new one is created
	Returns:
		Variable tensor
	'''
	with tf.device('/cpu:0'):
		dtype = tf.float16 if variable_float16 else tf.float32
		var =tf.get_variable(name, shape, initializer = initializer, dtype = dtype)
	return var

def altered_input():
	'''
	makes distorted input for training
	'''
	images, labels = inp.distort_input(
		batch_size = batch_size,
		data_dir = data_dir
	)
	if variable_float16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(images, tf.float16)
	return images, labels


def normal_input(iseval):
	'''
	gets normal inputs for training
	'''
	images, labels = inp.input(
		batch_size = batch_size,
		data_dir = data_dir,
		eval_data = not iseval
	)
	if variable_float16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(images, tf.float16)
	return images, labels


def conv(input,k_len,s_len,n_in,n_out,padding='SAME',scope='conv2d',apply_biases = False):
	'''
	Args:
		input: input into the layer
		k_len: kernel side length
		s_len: stride int the height and width direction
		n_in: depth of the input
		n_out: number of neurons
		padding: SAME vs VALID
		scope: name scope
		apply_biases: bool, whether to apply biases or not (redundant if beta 
			of batch normalization is used)
	Returns:
		output: preactivation convolution operation
	'''

	total_n_inputs = k_len*k_len*n_in
	stddev = math.sqrt(2.0/total_n_inputs)
		#from He et al.
	with tf.variable_scope(scope):

		kernel = variable(
			name = 'weight',
			shape = [k_len,k_len,n_in,n_out],
			initializer = tf.truncated_normal_initializer(stddev = stddev)
		)
		tf.add_to_collection('weights', kernel)
		strides = [1,s_len,s_len,1]
		conv = tf.nn.conv2d(
			input = input, 
			filter = kernel, 
			strides = strides,
			padding = padding
		)

		biases = variable(
			name = 'bias',
			shape = [n_out],
			initializer = tf.constant_initializer(0.001)
		)
		tf.add_to_collection('biases', biases)
		
		output = tf.nn.bias_add(conv, biases)

	return output

def batch_norm(x, n_in, istrain,scope = 'batch_norm'):
	'''
	applies batch normalization.
	
	the mean and var during test time are running averages collected during training

	Args:
		x: 4D inputs to batch_normalization node
		n_in: number of inputs into the normalization layer, used for gamma and beta
		scope: name scope
		istrain: whether on train or test
	Returns:
		out: normalized output
	'''
	with tf.variable_scope(scope):
		beta = variable(
			name = 'beta',
			shape = [n_in],
			initializer = tf.constant_initializer(0.0)
		)
		tf.add_to_collection('biases', beta)
		gamma = variable(
			name = 'gamma',
			shape = [n_in],
			initializer = tf.constant_initializer(1.0)
		)
		variance_epsilon = 1e-8
		mb_mean, mb_var = tf.nn.moments(x, [0,1,2])#mean and variance for a minibatch

		#use mean and variance of minibatch for training
		#use exponential moving average of means and variances of traing for the testing
		#	mean and variance
		#see:https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage 
		ema = tf.train.ExponentialMovingAverage(decay = EXPONETIAL_MOVING_AVERAGE_DECAY)
		def train_moving_average():
			maintain_averages_op = ema.apply([mb_mean, mb_var])
			with tf.control_dependencies([maintain_averages_op]):
				return tf.identity(mb_mean), tf.identity(mb_var)
		istrain = tf.cast(istrain,tf.bool)
		tf.cond(istrain,train_moving_average,
			lambda:(ema.average(mb_mean),ema.average(mb_var)))

		out = tf.nn.batch_normalization(
			x = x,
			mean = mb_mean,
			variance = mb_var,
			offset = beta,
			scale = gamma,
			variance_epsilon = variance_epsilon
		)
	return out #tf.reshape(out, x.get_shape())

def resnet_node(x, n_in, n_out, istrain,scope ='resnet_node'): # 3 Layered

	#single resnet block (bottleneck) that contains:
	#	- conv1: 1x1 convolution depth: n_out/4
	#	- conv2: 3x3 convolution depth: n_out/4
	#	- conv3: 1x1 convolution depth: n_out
	#Model: full pre-activation, see He et al. arxiv(1603.05027)
		
	#Args:
	#	x: input
	#	n_in: depth of input
	#	n_out: depth of output : min: 256, should be an exponential value of 2
	#	scope: name scope
	#Returns:
	#	out: pre-activation output

	with tf.variable_scope(scope):
		if n_in == n_out:
			shortcut = tf.identity(x, name = 'shorcut')
		else:
			shortcut =conv(#should be changed to linear projection of x to n_out depth
				input = x,
				k_len = 1,
				s_len = 1,  
				n_in=n_in, 
				n_out=n_out,
				scope = 'shortcut')

		#first conv
		#out = batch_norm(x = x, n_in = n_in, scope = 'batch_norm1',istrain=istrain)
		out = tf.nn.relu(x, name = 'activation1')
		out = conv(input = out,k_len = 1,s_len = 1, n_in=n_in, n_out=n_out/4,
			scope = 'conv1')

		#2nd conv
		#out = batch_norm(x = out, n_in = n_out/4, scope = 'batch_norm2',istrain=istrain)
		out = tf.nn.relu(out, name = 'activation2')
		out = conv(input = out,k_len = 3,s_len = 1, n_in=n_out/4, n_out=n_out/4,
			scope = 'conv2')

		#3rd conv
		#out = batch_norm(x = out, n_in = n_out/4, scope = 'batch_norm3',istrain=istrain)
		out = tf.nn.relu(out, name = 'activation3')
		out = conv(input = out,k_len = 1,s_len = 1, n_in=n_out/4, n_out=n_out,
			scope = 'conv3')

		
		out = out + shortcut

	return out


def resnet_layer(input, n_in, n_out, n, istrain, scope = 'resnet_layer'):
	with tf.variable_scope(scope):
		out = resnet_node(x = input, n_in = n_in, n_out = n_out, 
			scope ='resnet_node1', istrain = istrain)
		for i in range(1,n):
			out = resnet_node(x = out, n_in = n_out, n_out = n_out, 
				scope ='resnet_node%d'%(i+1), istrain = istrain)
	return out

def fully_con(x, n_in, n_out, scope = 'fully_connected'):
	with tf.variable_scope(scope):

		stddev = math.sqrt(2.0/n_in)
		weights = variable(
			name = 'weight', 
			shape = [n_in,n_out], 
			initializer = tf.truncated_normal_initializer(stddev = stddev)
		)
		tf.add_to_collection('weights', weights)
		biases = variable(
			name = 'bias',
			shape = [n_out],
			initializer = tf.constant_initializer(0.01)
		)
		tf.add_to_collection('biases', biases)
		out = tf.nn.relu(tf.matmul(x,weights)+biases)
	return out


def softmax_linear(x, n_out, scope='softmax'):
	with tf.variable_scope(scope):
		n_in = x.get_shape()[-1].value
		stddev = math.sqrt(2.0/n_in)

		weights = variable(name = 'weight', shape = [n_in,n_out], 
			initializer = tf.truncated_normal_initializer(stddev = stddev))
		tf.add_to_collection('weights', weights)

		biases = variable(name = 'bias',shape = [n_out],
			initializer = tf.constant_initializer(0.0))
		tf.add_to_collection('biases', biases)

		out = tf.add(tf.matmul(x,weights),biases)

	return out

def resnet(x,scope='resnet',istrain = False, own_im =False):
	with tf.variable_scope(scope):
		n_out = FIRST_CONV_LAYER_DEPTH
		out = conv(input = x,
			k_len = FIRST_CONV_LAYER_KERNEL_LENGTH,
			s_len = 1, n_in=N_CHANNELS, n_out=n_out,
			scope = 'init_conv')

		j = 1
		n_in = n_out
		n_out = START_DEPTH
		
		for i in RESNET_LAYER_INFO:
			out = resnet_layer(input = out, n_in = n_in, n_out = n_out, n = i, 
				scope = 'resnet_layer%d'%(j), istrain = istrain)
			j += 1
			n_in = n_out
			n_out = n_out*2
		

		out = batch_norm(x = out, n_in = out.get_shape()[-1], scope='Last_batch_norm',istrain=istrain)
		out = tf.nn.relu(out, name = 'Last_activation')
		out = tf.nn.avg_pool(value = out, ksize = [1,4,4,1],strides = [1,1,1,1],
			padding = 'SAME',name = 'pool')
		
		n_out = reduce(lambda x,y: x*y, out.get_shape().as_list()[1:-1])*START_DEPTH*2**(len(RESNET_LAYER_INFO)-1)
		if own_im:

			out = tf.reshape(out,[1,-1])
		else:
			out = tf.reshape(out,[batch_size,-1])
		
		out = fully_con(x = out,n_in = n_out, n_out = N_FC, scope = 'fully_connected1')
		out = softmax_linear(out, N_SML,scope = 'softmax1')

	return out
def label_smoothing(labels):
	'''
	defines label with label smoothing vectors
	Args:
		labels: vector of numbers from 0 to n_classes of size [batch_size]
	Returns:
		out: matrix of size [batch_size, n_classes]
	'''
	matrix = np.eye(N_CLASSES)*(1-2*OUTPUT_NOISE) + OUTPUT_NOISE
	with tf.device('/cpu:0'):
		params = tf.constant(matrix, name = 'params')
		out = tf.gather(params = params, indices = labels)
	return out

def loss(logits, labels, scope = 'loss'):
	'''
	applies softmax
	Args:
		logits: model output
		labels: real values
	Returns:
		loss: the softmax cross entropy of the loss
	'''
	with tf.variable_scope(scope):
		labels = tf.cast(labels, tf.int64)
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels = label_smoothing(labels),
			logits = logits,
			name = 'softmax_cross_entropy_loss'
		))
		tf.add_to_collection('losses', cross_entropy)

	return cross_entropy#tf.add_n(tf.get_collection('losses'),name = 'total_loss')
def accuracy(logits, labels, scope = 'accuracy'):
	with tf.variable_scope(scope):
		labels = tf.cast(labels, tf.int64)
		logits = tf.argmax(logits, axis = 1)
		out = tf.equal(labels, logits)
		out = tf.cast(out, tf.float32)
		return tf.reduce_mean(out)
def train(total_loss, global_step):
	'''
	Args: 
		total loss: loss from loss
		global_step: integer counting the number of the training steps
	Returns:
		out: operation for training

	'''
	opt = tf.train.AdamOptimizer(
		learning_rate = STEP_SIZE,
		beta1 = ADAM_BETA1,
		beta2 = ADAM_BETA2,
		epsilon = ADAM_DELTA,
		name = 'AdamOptimizer'
	)
	minimize_op = opt.minimize(total_loss,global_step = global_step)
	with tf.control_dependencies([minimize_op]):
		out = tf.no_op(name = 'train')
		return out


'''
Citations: 

@article{He2015,
	author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
	title = {Deep Residual Learning for Image Recognition},
	journal = {arXiv preprint arXiv:1512.03385},
	year = {2015}
}



'''
