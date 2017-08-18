import os

from six.moves import xrange
import tensorflow as tf
data_dir='/media/yukun/Barracuda Hard Drive 2TB/Data/CIFAR10/cifar-10-batches-bin'#Path to data directory
Sets = 'CIFAR10'
IMAGE_RESIZE =32
NUM_CLASSES= 10
N_EXAMPLES_TRAIN_EPOCH = 50000
N_EXAMPLES_EVAL_EPOCH =10000
num_tffile = 6
IMAGE_CUT_SIZE =24
num_threads = 8
Queue_Fraction = 0.4
def read_file(filename_queue):
	label_bytes = 1
	record_bytes = 32*32*3 + label_bytes#image data length + label data length
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	key, data = reader.read(filename_queue)
	record_bytes = tf.decode_raw(data, tf.uint8)
	label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]),tf.int32)
	image = tf.reshape(
		tf.strided_slice(record_bytes,[label_bytes],[label_bytes + 32*32*3]),
		[3, IMAGE_RESIZE, IMAGE_RESIZE]
	)
	image = tf.transpose(image, [1,2,0])
	return image, label

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
	images = tf.cast(images,tf.float32)
	return images, labels


def distort_input(batch_size, data_dir):
	filename_list = [os.path.join(
		data_dir,
		'data_batch_%d.bin'%i) for i in range(1,num_tffile)]
	filename_queue = tf.train.string_input_producer(filename_list)
	image, label = read_file(filename_queue)
	image = tf.cast(image, tf.float32)
	image = tf.image.resize_images(image, [IMAGE_RESIZE,IMAGE_RESIZE])
	image = tf.image.per_image_standardization(image)
	image = tf.image.random_flip_up_down(image)

	image = tf.random_crop(image, [IMAGE_CUT_SIZE, IMAGE_CUT_SIZE, 3])
	min_dequeue_examples = int(N_EXAMPLES_TRAIN_EPOCH*Queue_Fraction)

	label.set_shape([1])
	return batch_data(batch_size, image, label, min_dequeue_examples)



def input(batch_size, data_dir, eval_data = False, own_im = False):
	if not own_im:
		if eval_data:
			filename_list = [os.path.join(
				data_dir,
				'data_batch_%d.bin'%i) for i in range(1,num_tffile)]
		else:
			filename_list = [os.path.join(data_dir,'test_batch.bin')]
		
		filename_queue = tf.train.string_input_producer(filename_list)
		image, label = read_file(filename_queue)
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

	image = tf.cast(image, tf.float32)
	image = tf.image.resize_images(image, [IMAGE_RESIZE,IMAGE_RESIZE])
	image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_CUT_SIZE, IMAGE_CUT_SIZE)
	image = tf.image.per_image_standardization(image)
	min_dequeue_examples = int(N_EXAMPLES_EVAL_EPOCH*Queue_Fraction)
	image.set_shape([IMAGE_CUT_SIZE,IMAGE_CUT_SIZE,3])
	if own_im:
		return [image]
	label.set_shape([1])
	return batch_data(batch_size, image, label, min_dequeue_examples, shuffle = False,
		allow_smaller_final_batch = True)
