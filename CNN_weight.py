
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

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
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

Weight1 = weight_variable([3,3,3,32], "Weight1")
Weight1_mv = weight_variable([3,3,3,32], "Weight1_mv")
Weight1_zmv = weight_variable([3,3,3,32], "Weight1_zmv")

Weight2 = weight_variable([3,3,32,64], "Weight2")
Weight2_mv = weight_variable([3,3,32,64], "Weight2_mv")
Weight2_zmv = weight_variable([3,3,32,64], "Weight2_zmv")

Bias1	= tf.Variable(tf.zeros([32]))
Bias2	= tf.Variable(tf.zeros([64]))

x 			= tf.placeholder(tf.float32, shape = [None, dimension_input])
x_mv 		= tf.placeholder(tf.float32, shape = [None, dimension_input])
x_zmv 		= tf.placeholder(tf.float32, shape = [None, dimension_input])
target	 	= tf.placeholder(tf.float32, shape = [None, dimension_target])

x_image = tf.reshape(x, [-1, 8, 8, 3])
mv_image = tf.reshape(x_mv, [-1, 8, 8, 3])
zmv_image = tf.reshape(x_zmv, [-1, 8, 8, 3])
target_image = tf.reshape(target, [-1, 8, 8, 3])

# 1st layer
conv1 		= tf.nn.relu(conv2d(x_image,	Weight1) 		+ 	Bias1)
pool1 		= max_pool_2x2(conv1)
conv1_mv 	= tf.nn.relu(conv2d(mv_image,	Weight1_mv) 	+ 	Bias1)
pool1_mv 	= max_pool_2x2(conv1_mv)
conv1_zmv 	= tf.nn.relu(conv2d(zmv_image,	Weight1_zmv) 	+ 	Bias1)
pool1_zmv 	= max_pool_2x2(conv1_zmv)

# 2nd layer
conv2 		= tf.nn.relu(conv2d(pool1,	Weight2) 		+ 	Bias2)
pool2 		= max_pool_2x2(conv2)
conv2_mv 	= tf.nn.relu(conv2d(pool1_mv,	Weight2_mv) 	+ 	Bias2)
pool2_mv 	= max_pool_2x2(conv2_mv)
conv2_zmv 	= tf.nn.relu(conv2d(pool1_zmv,	Weight2_zmv) 	+ 	Bias2)
pool2_zmv 	= max_pool_2x2(conv2_zmv)

x_re = tf.reshape(pool2, [-1, 2*2*64])
xmv_re = tf.reshape(pool2_mv, [-1, 2*2*64])
xzmv_re = tf.reshape(pool2_zmv, [-1, 2*2*64])
#ref_x = np.hstack((xmv_re, xzmv_re))
#tmp_x = np.hstack((x_re, ref_x))

flat_x = tf.reshape([x_re, xmv_re, xzmv_re],[-1, 2*2*64*3])

#3rd layer
Weight3 = weight_variable([2*2*64*3,1024], "Weight3")
Bias3	= tf.Variable(tf.zeros([1024]))
lay3 = tf.nn.relu(tf.matmul(flat_x, Weight3) + Bias3)

#4th layer
Weight4 = weight_variable([1024,512], "Weight4")
Bias4	= tf.Variable(tf.zeros([512]))
lay4 = tf.nn.relu(tf.matmul(lay3, Weight4) + Bias4)

#5th layer
Weight5 = weight_variable([512,128], "Weight5")
Bias5	= tf.Variable(tf.zeros([128]))
lay5 = tf.nn.relu(tf.matmul(lay4, Weight5) + Bias5)

#out layer
Weight = weight_variable([128,3], "Weight")
Bias	= tf.Variable(tf.zeros([3]))
out_layer = tf.nn.softmax(tf.matmul(lay5, Weight) + Bias)

test = tf.concat([x,x_mv,x_zmv],1)
test_re = tf.reshape(test, [-1,192,3])
out_re = tf.reshape(out_layer, [-1,3,1])

out_image = tf.matmul(test_re,out_re)

out_img_re = tf.reshape(out_image, [-1,192])

'''
tr_x = reshape(total_x, [-1,16,16,1]

# 3rd layer 
Weight3 = weight_variable([5,5,1,32], "Weight3")
Bias3	= tf.Variable(tf.zeros([32]))
conv3 		= tf.nn.relu(conv2d(tr_x,Weight3) + Bias3)
pool3 		= max_pool_2x2(conv3)

# 4rd layer 
Weight4 = weight_variable([5,5,32,64], "Weight4")
Bias4	= tf.Variable(tf.zeros([64]))
conv4 		= tf.nn.relu(conv2d(pool3,Weight4) + Bias4)
pool4 		= max_pool_2x2(conv4)

# 4rd layer 
Weight4 = weight_variable([5,5,32,64], "Weight4")
Bias4	= tf.Variable(tf.zeros([64]))
conv4 		= tf.nn.relu(conv2d(pool3,Weight4) + Bias4)
pool4 		= max_pool_2x2(conv4)
'''

lr = 0.0001
'''
MSE = 0
for i in range(64):
	MSE += (outlayer[i] - target[i])*(outlayer[i] - target[i])
PSNR = tf.reduce_mean(10.0 * log10(255.0*255.0 / (MSE/64.0)))
'''
loss = tf.reduce_mean(tf.square(out_img_re - target))
#optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
init_op = tf.global_variables_initializer()

sess = tf.InteractiveSession()
saver = tf.train.Saver()

#saver.restore(sess, "tmp/model8_adam/model.ckpt")
sess.run(init_op)
for i in range(10000):
	#if i%4000 == 0:
		#Data_batch, mv_batch, zmv_batch, Target_batch = next_batch(40000, Training.input, T_mv.input, T_zmv.input, Target.input)
	if i%1000 == 0:
		print("step %d, loss %g"%(i, loss.eval(feed_dict={
			x		:Training.input,
			x_mv	:T_mv.input,
			x_zmv	:T_zmv.input,
			target	:Target.input}
			)))
	optimizer.run(feed_dict={
			x		:Training.input,
			x_mv	:T_mv.input,
			x_zmv	:T_zmv.input,
			target	:Target.input})
	
#print("final loss %g" % loss.eval(feed_dict={x: input_data.test, target_x:input_data.target}))

output_nd = out_image.eval(feed_dict = {
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




