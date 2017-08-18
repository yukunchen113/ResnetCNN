'''
This reads images from caltech256 and makes them into tfrecords file.
'''
import tensorflow as tf 
import numpy as np 

from PIL import Image 

import os
import shutil

main_dir='/media/yukun/Barracuda Hard Drive 2TB/Data/Caltech256'#The main directory to work in
num_tffile=5#The number of tfrecord files to create. One of the files are for testing

def get_image(filename):
	'''
	returns the image data for the filename.
	Args: 
		filepath of an image
	Returns:
		image: in string format,
		shape: [height, width, depth] is an string
	'''
	image = Image.open(filename)
	image = np.asarray(image, np.uint8)
	try:#converts grayscale to RGB
		image.shape[2]
	except IndexError:
		image = np.transpose(np.multiply(image, np.ones((3, image.shape[0],image.shape[1]))
			),(1,2,0)).astype(np.uint8)
	shape = image.shape

	return image.tostring(), shape#np.asarray(shape).astype(np.int64).tostring()

def readimages():
	'''
	Goes through every image and gets the image, putting them into an array
	Args:
		None
	Returns:
		data: array of information including:
			image: string
			shape: int 
			label: index from 0 to 256 for the classes
	'''
	data = []
	j = 0
	object_dir = os.path.join(main_dir, '256_ObjectCategories')
	objectCategories = os.listdir(object_dir)
	for i in range(len(objectCategories)):
		label = i
		examp_dir = os.path.join(object_dir, objectCategories[i])
		examples = os.listdir(examp_dir)
		print 'compiling data: %d/%d'%(i+1, len(objectCategories))
		for examp in examples:
			image_dir = os.path.join(examp_dir, examp)
			image, shape = get_image(image_dir)
			try:
				data[j]
			except IndexError:
				data.append([])

			data[j].append([image, shape, label])
			j = (j+1)%num_tffile
	return data

def make_bytes(item):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value = [item]))

def make_int64(item):
	return tf.train.Feature(int64_list=tf.train.Int64List(value = [item]))

def write_data(data):
	'''
	writes the data into tfrecord files
	Args:
		data:
			of size [num_tffile, total_examples/num_tffile, 3] the 3 at the end is for:
				[image, shape, label]
	Returns:
		nothing
	'''
	write_dir = os.path.join(main_dir, 'caltech-256-batches-bin')
	if os.path.exists(write_dir):
		shutil.rmtree(write_dir)
	os.mkdir(write_dir)
	for i in range(len(data)):
		if i:
			filename = os.path.join(write_dir, 'train_batch_%d.tfrecords'%(i))
		else:
			filename = os.path.join(write_dir, 'test_batch.tfrecords')
		print 'making files %d of %d'%(i+1, len(data))
		with tf.python_io.TFRecordWriter(filename) as writer:
			for j in range(len(data[i])):
				image,shape,label = data[i][j]
				height, width, depth = shape
				writer.write(tf.train.Example(features=tf.train.Features(feature={
					'image':make_bytes(image),
					'height':make_int64(height),
					'width':make_int64(width),
					'depth':make_int64(depth),
					'label':make_int64(label)
					})).SerializeToString())

def main(argv=None):
	data = readimages()
	write_data(data)

if __name__ == '__main__':
	tf.app.run()
