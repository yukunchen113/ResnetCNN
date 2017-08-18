import tensorflow as tf 
import numpy as np 
import os
import model
import train
from PIL import Image
import caltech256_input as inp
train_dir = train.train_dir
'''---caltech

import caltech256_bin as bina
main_dir = bina.main_dir
img_dir = os.path.join(main_dir,'256_ObjectCategories')
image = inp.input(1,'a',own_im = True)
logit = model.resnet(image,scope = 'resnet1', own_im=True)
guess = tf.argmax(logit, axis = 1)
checkpt = tf.train.get_checkpoint_state(train_dir)
saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess, checkpt.model_checkpoint_path)
	[guess] = sess.run(guess)
	print 'Idk man, is it a ' + os.listdir(img_dir)[guess].split('.')[-1]
'''

#cifar10
own_im_path = '/home/yukun/Software/Python/MyCode/CNN/images'
IMG_SIZE = 32
IMAGE_CUT_SIZE = 24
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
image = tf.image.resize_images(image, [IMG_SIZE,IMG_SIZE])
image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_CUT_SIZE, IMAGE_CUT_SIZE)
image.set_shape([IMAGE_CUT_SIZE,IMAGE_CUT_SIZE,3])
image = [image]
logit = model.resnet(image,scope = 'resnet1', own_im=True)
guess = tf.argmax(logit, axis = 1)
checkpt = tf.train.get_checkpoint_state(train_dir)
saver = tf.train.Saver()
lista = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
with tf.Session() as sess:
	saver.restore(sess, checkpt.model_checkpoint_path)
	[guess] = sess.run(guess)
	print 'Idk man, is it a ' + lista[guess] + '?'