from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf 
import numpy as np 

import os
import shutil
import model
Sets = model.Sets
batch_size= model.batch_size

write_frequency=10#the frequency of writing the training information after n steps
train_dir='/media/yukun/Barracuda Hard Drive 2TB/Data/'+ Sets +'/Data' #Path to train directory
max_steps=100000#Number of batches to run


def train():
	with tf.Graph().as_default():
		global_step = tf.contrib.framework.get_or_create_global_step()
		with tf.device('/cpu:0'):
			images, labels = model.altered_input()
		logits = model.resnet(images, scope = 'resnet1', istrain = True)
		loss = model.loss(logits, labels, scope = 'loss1')
		accuracy = model.accuracy(logits,labels, scope = 'accuracy')
		train_op = model.train(loss, global_step)
		class Log(tf.train.SessionRunHook):
			def begin(self):
				self.step = -1
				self.start_time = time.time()
				self.total_time = time.time()

			def before_run(self, run_context):
				self.step += 1
				return tf.train.SessionRunArgs([loss, accuracy])

			def after_run(self, run_context, run_values):
				'''
				logs loss, examples per second, seconds per batch
				'''
				if not self.step%write_frequency:
					curtime = time.time()#current time
					duration = curtime - self.start_time#start time
					total_dur = curtime - self.total_time
					ts = total_dur%60
					tm = (total_dur//60)%60
					th = total_dur//3600
					self.start_time = curtime
					[loss, accuracy] = run_values.results
					ex_per_sec = write_frequency*batch_size/duration
					sec_per_batch = float(duration/write_frequency)

					str = ('step: %d, accuracy:%.3f loss: %.3f, examples/sec: %.2f, sec/batch: %1f, total time: %dh %dm %ds')
					print(str%(self.step,accuracy,loss,ex_per_sec,sec_per_batch, th,tm,ts))
		with tf.train.MonitoredTrainingSession(
			checkpoint_dir = train_dir,#for checkpoint writing
			
			hooks = [#things to do while running the session
				tf.train.StopAtStepHook(last_step = max_steps),
				tf.train.NanTensorHook(loss),
				Log()],
				save_summaries_steps = 100
		) as sess:
			while not sess.should_stop():
				sess.run(train_op)
				
def main(argv=None):
	train()
if __name__ == '__main__':
	#removes checkpoint directory (should only be used if model was changed)
	if os.path.exists(train_dir):
		shutil.rmtree(train_dir)	
	os.mkdir(train_dir)
	
	tf.app.run()
