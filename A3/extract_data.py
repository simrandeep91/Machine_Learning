from __future__ import division
from __future__ import print_function
import os
import glob
import sys
import gzip
import tensorflow as tf;
import numpy as np
from PIL import Image
from numpy import dtype
from _ast import Num
from matplotlib.pyplot import imshow

def extract_data2(num_images=7000, num_pixels=3072):
  #num_images = 10
  path='/home/simrandeep/workspace/cs411_a3/csc411-a3/train'
  jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))
  input_data = np.zeros(shape=(num_images,num_pixels))
  #print (input_data.shape)

  for filename in jpg_files_path:
    index = int(filename[-9:-4].lstrip("0"))
    #3-D
    im = Image.open(filename)
    #print(im.size)
    #im.show()
    #print(im.size)
    #im.show();
    #GreyScale
    #im = Image.open(filename).convert('LA')
    
    #Black and White
    #im = Image.open(filename).convert('1')
    im = im.resize((32,32),Image.NEAREST)
    #print("shape")
    #print(list(im.getdata()))
    data = np.asarray(im).astype(np.int32)
    #print(data.shape)
    h,w,c = data.shape
    data = data.reshape(h*w*c)
    #print(data)
    #print (data, data.shape)
    #data = (data - (255 / 2.0)) / 255
    data = (data) / 255
    #print(data)
    #print (data)
    input_data[index-1]=np.copy(data)
    #print (input_data[index])
  return input_data


def extract_data_alex(num_images=7000, num_pixels=227*227*3):
  path='./train/'
  jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))
  input_data = np.zeros(shape=(num_images,227,227,3))

  for filename in jpg_files_path:
    index = int(filename[-9:-4].lstrip("0"))
    #3-D
    im = Image.open(filename)
    #GreyScale
    #im = Image.open(filename).convert('LA')
    #Black and White
    #im = Image.open(filename).convert('1')
    im = im.resize((227,227),Image.ANTIALIAS)
    #im = im[:,:,::-1]
    #im = im.transpose((2,0,1))
    data = np.asarray(im).astype(np.float32)
    #h,w,c = data.shape
    #data = data.reshape(h*w*c)
    #data = (data) / 255
    input_data[index-1]=np.copy(data)
  return input_data
  

def extract_data_alex_valid(num_images=970, num_pixels=227*227*3):
  path='./val/'
  jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))
  input_data = np.zeros(shape=(num_images,227,227,3))

  for filename in jpg_files_path:
    index = int(filename[-9:-4].lstrip("0"))
    #3-D
    im = Image.open(filename)
    #GreyScale
    #im = Image.open(filename).convert('LA')
    #Black and White
    #im = Image.open(filename).convert('1')
    im = im.resize((227,227),Image.ANTIALIAS)
    #im = im[:,:,::-1]
    #im = im.transpose((2,0,1))
    data = np.asarray(im).astype(np.float32)
    #h,w,c = data.shape
    #data = data.reshape(h*w*c)
    #data = (data) / 255
    input_data[index-1]=np.copy(data)
  return input_data


def extract_data2_pca(num_images=7000, num_pixels=3072):
  path='/home/simrandeep/workspace/cs411_a3/csc411-a3/train_small'
  jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))
  input_data = np.zeros(shape=(num_images,num_pixels))

  immatrix = np.asarray([np.array(Image.open(filename)).flatten() for filename in jpg_files_path],'f')
  V,S,immean = pca_img.pca(immatrix)
  print(V.shape)
  print(S.shape)
  print(immean.shape)
  return V



def extract_data2_valid(num_images=970, num_pixels=3072):
  #num_images = 10
  path='/home/simrandeep/Downloads/411_A3/val/'
  jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))
  input_data = np.zeros(shape=(num_images,num_pixels))
  for filename in jpg_files_path:
    index = int(filename[-9:-4].lstrip("0"))
    #3-D
    im = Image.open(filename)
    #print(im.size)
    #im.show()
    #print(im.size)
    #im.show();
    #GreyScale
    #im = Image.open(filename).convert('LA')
    
    #Black and White
    #im = Image.open(filename).convert('1')
    im = im.resize((32,32),Image.NEAREST)
    #print("shape")
    #print(list(im.getdata()))
    data = np.asarray(im).astype(np.int32)
    #print(data.shape)
    h,w,c = data.shape
    data = data.reshape(h*w*c)
    #print(data)
    #print (data, data.shape)
    #data = (data - (255 / 2.0)) / 255
    data = (data) / 255
    #print(data)
    #print (data)
    input_data[index-1]=np.copy(data)
    #print (input_data[index])
  return input_data


def extract_label(num_output = 8):
    r = np.genfromtxt('/home/ubuntu/csc411-a3/train.csv', delimiter=',', dtype=None, names=True)
    #print(r.shape)
    #print(r)
    t = r['Label']
    target = np.zeros((r.shape[0],num_output));
    print(target.shape)
    for i in range(target.shape[0]):
        #print(t[i])
        target[i,t[i]-1] = 1;
    #print(target)
    return target

def extract_label_valid(num_output = 8):
    r = np.genfromtxt('/home/simrandeep/Downloads/411_A3/sample_submission.csv', delimiter=',', dtype=None, names=True)
    #print(r.shape)
    #print(r)
    t = r['Prediction']
    target = np.zeros((r.shape[0],num_output));
    print(target.shape)
    for i in range(target.shape[0]):
        #print(t[i])
        target[i,t[i]-1] = 1;
    #print(target)
    return target

def extract_label2(num_output = 8):
    r = np.genfromtxt('/home/ubuntu/csc411-a3/train.csv', delimiter=',', dtype=None, names=True)
    #print(r.shape)
    #print(r)
    target = r['Label']
    return target

def extract_label2_valid(num_output = 8):
    r = np.genfromtxt('/home/simrandeep/Downloads/411_A3/sample_submission.csv', delimiter=',', dtype=None, names=True)
    #print(r.shape)
    #print(r)
    target = r['Prediction']
    return target

def main(is_directory=True):
  """Reads directory of images in tensorflow
  Args:
    path:
    is_directory:

  Returns:
  
  """  
  input_data = extract_data_alex(1)
  y = extract_label()
  print(input_data)
  print(y[1:5])
  
  x = tf.placeholder("float", shape=[None, 128*128*3])
  y_ = tf.placeholder("float", shape=[None, 8])
  
  
  return

def extra():  
  path = "/home/vd/csc411-a3/train.gz";
  images = []
  jpeg_files = []
  
  #extract_data(path, 1)

  reader = tf.WholeFileReader()

  jpeg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][eE][gG]'))
  jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))

  if is_directory:
    for filename in jpeg_files_path:
      jpeg_files.append(filename)
    for filename in jpg_files_path:
      jpeg_files.append(filename)
  else:
    raise ValueError('Currently only batch read from directory supported')

 # print(jpeg_files[0]);
  if len(jpeg_files) > 0:
    jpeg_file_queue = tf.train.string_input_producer(jpeg_files)
    jkey, jvalue = reader.read(jpeg_file_queue)
    j_img = tf.image.decode_jpeg(jvalue)
  
  #print(jpeg_file_queue.size)
  #print(jvalue)
  #print(jkey)
  #print(jvalue)
  #print(j_img)
    
  init_op = tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init_op)
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    imshape = [128,128,3]
    ip = np.zeros(shape=(10,49152),dtype=object);
    for i in range(10): #length of your filename list
      image = j_img.eval() #here is your image Tensor :)
      #assert image.shape[3] == 1
      image = image.reshape(10, 128 * 128)
        # Convert from [0, 255] -> [0.0, 1.0].
      image = image.astype(np.float32)
      image = np.multiply(image, 1.0 / 255.0)
      print(image)
#       num_elements = 1
#       for i in imshape: 
#           num_elements = num_elements * i
#       print (num_elements)
#       image = tf.reshape(image, [num_elements])
#       image.set_shape(num_elements)
#       print (image)
      #ip[i,0] = image
    coord.request_stop()
    coord.join(threads)
    #print(ip)

if __name__ == '__main__':
  tf.app.run()
