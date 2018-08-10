
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import Training as Training
import Training_mv as T_mv
import Training_zmv as T_zmv
import Testing as Testing
import Testing_mv as Testing_mv
import Testing_zmv as Testing_zmv
import Target as Target
import os
import pickle
import time

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.1), name)
def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape = shape), name)
	
def next_batch(num, data, data_mv, data_zmv, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    mv_shuffle = [data_mv[i] for i in idx]
    zmv_shuffle = [data_zmv[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(mv_shuffle), np.asarray(zmv_shuffle), np.asarray(labels_shuffle)

tStart = time.time()
	
tf.reset_default_graph()

dimension_input 	= 192
dimension2			= 512
dimension3			= 512
dimension_target  	= 192

Weight1 = weight_variable([dimension_input, dimension2], "Weight1")
Weight1_mv = weight_variable([dimension_input, dimension2], "Weight1_mv")
Weight2_zmv = weight_variable([dimension_input, dimension2], "Weight1_zmv")

Weight2 = weight_variable([dimension2, dimension3], 	 "Weight2")
Weight8 = weight_variable([dimension2, dimension_target],  "Weight8")

Bias1	= tf.Variable(tf.zeros([dimension2]))
Bias2	= tf.Variable(tf.zeros([dimension3]))
Bias8	= tf.Variable(tf.zeros([dimension_target]))

x 			= tf.placeholder(tf.float32, shape = [None, dimension_input])
x_mv 		= tf.placeholder(tf.float32, shape = [None, dimension_input])
x_zmv 		= tf.placeholder(tf.float32, shape = [None, dimension_input])
target	 	= tf.placeholder(tf.float32, shape = [None, dimension_target])

layer1 		= tf.nn.relu(tf.matmul(x, Weight1) + tf.matmul(x_mv, Weight1_mv) + tf.matmul(x_zmv, Weight2_zmv) + Bias1 )
layer2 		= tf.nn.relu(tf.matmul(layer1, Weight2) + Bias2)
outlayer 	= tf.nn.relu(tf.matmul(layer2, Weight8) + Bias8)

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.0001, 40000, 10000, 0.96, staircase=True)

MSE = 0
for i in range(64):
	MSE += (outlayer[i] - target[i])*(outlayer[i] - target[i])
PSNR = tf.reduce_mean(10.0 * log10(255.0*255.0 / (MSE/64.0)))

#loss = tf.reduce_mean(tf.square(outlayer - target))
#optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss)
optimizer = tf.train.AdamOptimizer(lr).minimize(PSNR)
init_op = tf.global_variables_initializer()

sess = tf.InteractiveSession()
saver = tf.train.Saver()

#saver.restore(sess, "tmp/model8_adam/model.ckpt")
sess.run(init_op)
for i in range(20000):
	if i%4000 == 0:
		Data_batch, mv_batch, zmv_batch, Target_batch = next_batch(40000, Training.input, T_mv.input, T_zmv.input, Target.input)
	if i%1000 == 0:
		print("step %d, learning rate",i, lr.eval())
		'''
		print("step %d, PSNR %g"%(i, PSNR.eval(feed_dict={
			x		:Data_batch,
			x_mv	:mv_batch,
			x_zmv	:zmv_batch,
			target	:Target_batch}
			)))
		'''
	optimizer.run(feed_dict={
			x		:Data_batch,
			x_mv	:mv_batch,
			x_zmv	:zmv_batch,
			target	:Target_batch})
	
#print("final loss %g" % loss.eval(feed_dict={x: input_data.test, target_x:input_data.target}))

output_nd = outlayer.eval(feed_dict = {
			x		:Testing.input,
			x_mv	:Testing_mv.input,
			x_zmv	:Testing_zmv.input,
			target	:Target.input})

#save_path = saver.save(sess, "tmp/model8_adam/model.ckpt") 

f=open('Output.txt','w')
f.close()
f=open('Output.txt','ab')

for s in output_nd:
	np.savetxt(f, s,fmt='%f')

f.close()

tEnd = time.time()
print (tEnd-tStart)

os.system("pause")




