
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf 
import numpy as np
from six.moves import xrange
from PIL import Image
import os

import caltech256_bin
main_dir=caltech256_bin.main_dir
num_tffile =caltech256_bin.num_tffile
data_dir='/media/yukun/Barracuda Hard Drive 2TB/Data/Caltech256/caltech-256-batches-bin'#The data directory
own_im_path = '/home/yukun/Software/Python/MyCode/CNN/images'
Sets = 'Caltech256'

IMAGE_RESIZE = 32#this is to resize the image
IMAGE_CUT_SIZE = 24#image is cut down to this size with a random crop or a normal one
	#should be more than 1/2 of image resize
NUM_CLASSES = 257
N_EXAMPLES_TRAIN_EPOCH = 24485
N_EXAMPLES_EVAL_EPOCH =6131
num_threads = 8
Queue_Fraction = 0.2

def read_file(filename_queue):
	reader = tf.TFRecordReader()
	key, serialized_data = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_data,
		features = {
			'image': tf.FixedLenFeature([], tf.string),
			'height': tf.FixedLenFeature([], tf.int64),
			'width': tf.FixedLenFeature([], tf.int64),
			'depth': tf.FixedLenFeature([], tf.int64),
			'label': tf.FixedLenFeature([], tf.int64)
		})
	image = tf.decode_raw(features['image'],tf.uint8)
	label = tf.cast(features['label'], tf.int32)
	height = tf.cast(features['height'], tf.int32)
	width = tf.cast(features['width'], tf.int32)
	depth = tf.cast(features['depth'], tf.int32)
	shape = [height, width, depth]
	image = tf.reshape(image, shape)
	return image, label, shape


def batch_data(batch_size, image, label, min_dequeue_examples, shuffle = True, allow_smaller_final_batch= False):
	
	capacity = min_dequeue_examples + 3*batch_size
	if shuffle:
		images, labels = tf.train.shuffle_batch(
			[image, label],
			min_after_dequeue = min_dequeue_examples, 
			capacity = capacity,
			batch_size = batch_size,
			num_threads = num_threads,
			allow_smaller_final_batch = allow_smaller_final_batch)
		return images, labels
	images, labels = tf.train.batch(
		[image, label],
		capacity = capacity,
		batch_size = batch_size,
		num_threads = num_threads,
		allow_smaller_final_batch = allow_smaller_final_batch
		)
	return images, labels
def square_image(image, shape):
	'''
	makes image to a square of size IMAGE_RESIZE

	'''
	
	biggest_side=tf.cond(tf.less(shape[0], shape[1]), lambda: shape[1], lambda:shape[0])
	biggest_side = tf.cast(biggest_side, tf.int32)
	print(image.get_shape())
	image = tf.image.resize_image_with_crop_or_pad(image, biggest_side, biggest_side)
	image = tf.image.resize_images(image, [IMAGE_RESIZE, IMAGE_RESIZE],
		method = tf.image.ResizeMethod.BICUBIC)
	return image


def distort_input(batch_size, data_dir):
	filename_list = [os.path.join(
		data_dir,
		'train_batch_%d.tfrecords'%i) for i in range(1,num_tffile)]
	filename_queue = tf.train.string_input_producer(filename_list)
	image, label, shape = read_file(filename_queue)
	#image = square_image(image, shape)
	image = tf.image.resize_images(image, [IMAGE_RESIZE,IMAGE_RESIZE])
	image = tf.image.per_image_standardization(image)
	image = tf.image.random_flip_up_down(image)

	image = tf.random_crop(image, [IMAGE_CUT_SIZE, IMAGE_CUT_SIZE, 3])
	min_dequeue_examples = int(N_EXAMPLES_TRAIN_EPOCH*Queue_Fraction)
	return batch_data(batch_size, image, label, min_dequeue_examples)



def input(batch_size, data_dir, eval_data = False, own_im = False):
	if not own_im:
		if eval_data:
			filename_list = [os.path.join(
				data_dir,
				'train_batch_%d.tfrecords'%i) for i in range(1,num_tffile)]
		else:
			filename_list = [os.path.join(data_dir,'test_batch.tfrecords')]
		
		filename_queue = tf.train.string_input_producer(filename_list)
		image, label, shape = read_file(filename_queue)
	if own_im:
		image_name = os.listdir(own_im_path)[0]
		image_path = os.path.join(own_im_path, image_name)
		image = Image.open(image_path)
		image = np.asarray(image, np.uint8)
		try:
			image.shape[2]
		except IndexError:
			image = np.transpose(np.multiply(image, np.ones((3, image.shape[0],image.shape[1]))
				),(1,2,0)).astype(np.uint8)
		shape = image.shape
		image = tf.cast(image, tf.uint8)
	image = square_image(image, shape)
	image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_CUT_SIZE, IMAGE_CUT_SIZE)
	min_dequeue_examples = int(N_EXAMPLES_EVAL_EPOCH*Queue_Fraction)
	image.set_shape([IMAGE_CUT_SIZE,IMAGE_CUT_SIZE,3])
	if own_im:
		return [image]
	return batch_data(batch_size, image, label, min_dequeue_examples, shuffle = False,
		allow_smaller_final_batch = True)
