from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from extract_data import *
from PIL import Image
from numpy import random
from utils import *

import tensorflow as tf
import csv

from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 8))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]



################################################################################
#Read Image


#im1 = (imread("00001.jpg")[:,:,:3]).astype(float32)
# im1 = Image.open("00001.jpg")
# im1 = im1.resize((227,227),Image.ANTIALIAS)
# im1 = np.asarray(im1).astype(np.float32)
# im1 = im1 - mean(im1)
# 
# #im2 = (imread("00002.jpg")[:,:,:3]).astype(float32)
# im2 = Image.open("00002.jpg")
# im2 = im2.resize((227,227),Image.ANTIALIAS)
# im2 = np.asarray(im2).astype(np.float32)
# im2 = im2 - mean(im2)

learning_rate = 0.00001
training_iters = 650
batch_size = 64
display_step = 1

# Network Parameters
n_input = 32*32*3 # MNIST data input (img shape: 28*28)
n_classes = 8 # MNIST total classes (0-9 digits)
dropout = 0.50 # Dropout, probability to keep units

inputs_t = extract_data_alex(7000)
data = inputs_t[:,:];

print(data.shape)

inputs_v = extract_data_alex_valid(970)
data_valid = inputs_v[:,:];

print(data_valid.shape)

tt = extract_label();
label = tt[:,:];
print(label.shape)

#label_valid = tt[6500:7000,:];

b = extract_label2();
batch_result_1 = b;

#batch_result_valid = b[6500:7000]

#n = np.concatenate((data,label,batch_result_1),axis=1)
#a = np.random.shuffle(n)
#print(a.shape)

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))


net_data = load("bvlc_alexnet.npy").item()
#print(net_data["fc8"][0])

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



x = tf.placeholder(tf.float32, (None,) + xdim)
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)


print("conv1")
print(conv1)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')

#lrn(2, 2e-02, 0.50, name='norm1')



radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


print("conv2")
print(conv2)



#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')

#lrn(2, 2e-02, 0.50, name='norm1')



radius = 2; alpha = 2e-07; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)



print("conv3")
print(conv3)



#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)




print("conv4")
print(conv4)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)



print("conv5")
print(conv5)


#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#maxpool7 = tf.nn.max_pool(maxpool6, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
#print(maxpool7.get_shape())
#fc6
#fc(4096, name='fc6')

#fc6W = tf.Variable(net_data["fc6"][0][:,:8])
#fc6b = tf.Variable(net_data["fc6"][1][:8])
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
#print("here")
#print(fc6W.get_shape())
#print(fc6b.get_shape())
#print(int(prod(maxpool5.get_shape()[1:])))
#print(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]))
fc6 = tf.nn.relu_layer((tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))])), fc6W, fc6b)
#print(fc6.get_shape())
#fc7
#fc(4096, name='fc7')
#fc7W = tf.Variable(net_data["fc7"][0][:,:8])
#fc7b = tf.Variable(net_data["fc7"][1][:8])
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])

fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
#print(fc7)
#fc8
#fc(1000, relu=False, name='fc8')
#print(net_data["fc8"][0])
#fc8W = tf.Variable(net_data["fc8"][0][:,8:16])
#fc8b = tf.Variable(net_data["fc8"][1][8:16])


#stddev=.25
fc8W = tf.Variable(tf.random_normal([4096, 8], stddev=0.35),name="fc8W")
fc8b = tf.Variable(tf.zeros([8]), name="fc8b")
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prob, y))
cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(prob, y))))

#momentum = 0.5

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# 
# # Evaluate model
correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init = tf.initialize_all_variables()
pro = np.zeros((batch_result_1.shape[0]));
pro_valid = np.zeros((data_valid.shape[0]));

print("pro valid shape")
print(pro_valid.shape)

# 
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations
    for i in range(0,30):
        step = 0
        while step * batch_size < 7000:
            start = step * batch_size
            end = min(7001, (step + 1) * batch_size)
            batch_x = data[start:end]
            batch_y = label[start:end]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            pr = sess.run(prob, feed_dict={x: batch_x})
            pro[start:end] = print_prob(pr, './caffe_classes.txt', batch_result_1[start:end])

            step += 1
        print("Optimization Finished!")
        print("###############################")
        print("###############################")
        # test save
        cl = 0;
        for k in range(0, pro.shape[0]):
            if pro[k] == batch_result_1[k]:
                cl = cl+1
 
        cl = (cl*1.0)/pro.shape[0]
        print("TRAINING DATASET Accuracy")
        print(cl)

        step1=0
        while step1 * batch_size < 970:
            start_v = step1 * batch_size
            end_v = min(971, (step1 + 1) * batch_size)
            batch_valid_x = data_valid[start_v:end_v]
            pr_valid = sess.run(prob, feed_dict={x: batch_valid_x})
            pro_valid[start_v:end_v] = guess_print_prob(pr_valid, './caffe_classes.txt')
            step1+=1;

        print(pro_valid)

        print("###########################")
        print("###########################")


    with open('test.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        for i in range(2971):
            a.writerow([i]);

    with open('test.csv','r') as csvinput:
        with open('output.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)

            all = []
            row = next(reader)
            row.append('Prediction')
            all.append(row)
            i = 0;
            for row in reader:
                if(i<970):
                    row.append(pro_valid[i])
                else:
                    row.append("0")
                all.append(row)
                i = i+1;

            writer.writerows(all)
 
#        cl_v = 0;
#        for k in range(0, pro_valid.shape[0]):
#            if pro_valid[k] == batch_result_valid[k]:
#                cl_v = cl_v+1
#
#        cl_v = (cl_v*1.0)/pro_valid.shape[0]
#        print("VALIDATION DATASET Accuracy")
#        print(cl_v)
#        print(pro_valid)


