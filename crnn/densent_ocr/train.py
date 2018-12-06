# -*- coding: utf-8 -*-
import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')
from io import open
from keras.layers.convolutional import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Masking,Lambda,Permute
from keras.layers import Input,Dense,Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import GRU,LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam,SGD,Adadelta
from keras import losses
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.utils import plot_model
import tensorflow as tf  

import numpy as np 
import os
from PIL import Image 
import json
import threading

from imp import reload 
import densenet
reload(densenet)


def get_session(gpu_fraction=0.6):  
    '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''  
  
    num_threads = os.environ.get('OMP_NUM_THREADS')  
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)  
  
    if num_threads:  
        return tf.Session(config=tf.ConfigProto(  
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))  
    else:  
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
  
K.set_session(get_session()) 



def ctc_lambda_func(args):
    y_pred,labels,input_length,label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


char=''
#with open('/datasets/TextRecognitionDataGenerator/TextRecognitionDataGenerator/dicts/cn.txt',encoding='utf-8') as f:
with open('../../../text_renderer-dev/data/chars/med.txt',encoding='utf-8') as f:
      for ch in f.readlines():
            ch = ch.strip()
            char=char+ch
            
#caffe_ocr中把0作为blank，但是tf 的CTC  the last class is reserved to the blank label.
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/ctc/ctc_loss_calculator.h
#char =char[1:]+'卍'
char =char+'卍'
print('nclass:',len(char))

id_to_char = {i:j for i,j in enumerate(char)}
#print(id_to_char[5988])

maxlabellength = 20
img_h = 32
img_w = 280
nclass = len(char)
old_nclass = 346
rnnunit=256
batch_size =64

    
class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self,total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self,batchsize):
        r_n=[]
        if(self.index+batchsize>self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index+batchsize)-self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
            
        else:
            r_n = self.range[self.index:self.index+batchsize]
            self.index = self.index+batchsize
        return r_n  
    
def readtrainfile(filenamelist):
    dic={}
    for filename in filenamelist:
        if 'tmp_labels' in filename:
            DATA_DIR = os.path.dirname(filename)
        elif 'TextRecognitionDataGenerator' in filename:
            DATA_DIR = filename.split('.')[0]    
        #DATA_DIR = os.path.join('/home/ls/project201803/tensorflow.image/datasets/TextRecognitionDataGenerator/TextRecognitionDataGenerator/',
        #                        os.path.basename(filename).split('.')[0]) 
        with open(filename,'r', encoding='utf-8') as f:
            lines = f.readlines() 
            for i in lines:
                p = i.split(' ')
                p[0] = os.path.join(DATA_DIR, p[0])
                dic[p[0]] = p[1:]
    return dic

def gen3(trainfilelist,batchsize=64,maxlabellength=10,imagesize=(32,280)):
    image_label = readtrainfile(trainfilelist)
    _imagefile = [i for i,j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize,maxlabellength])*10000
    input_length = np.zeros([batchsize,1])
    label_length = np.zeros([batchsize,1])
    
    r_n = random_uniform_num(len(_imagefile))
    print '图片总量',len(_imagefile)
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i,j in enumerate(shufimagefile):
            img1 = Image.open(j).convert('L')
            # resize image to imagesize
            original_w, original_h = img1.size
            if original_w > imagesize[1] or original_h > imagesize[0]:
                print 'error size'
            img = np.array(img1,'f')/255.0-0.5
            pad = imagesize[1] - original_w
            pad_0, pad_1 = int(pad / 2), pad - int(pad / 2)
            img = np.lib.pad(img, ((0, 0), (pad_0, pad_1)), 'constant', constant_values=(-0.5))

            x[i] = np.expand_dims(img,axis=2)
            #print('imag:shape',img.shape)
            str = image_label[j]
            label_length[i] = len(str) 
            
            if(len(str)<=0):
                print("len<0",j)
            input_length[i] = imagesize[1]//8
            #caffe_ocr中把0作为blank，但是tf 的CTC  the last class is reserved to the blank label.
            #https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/ctc/ctc_loss_calculator.h
            #
            labels[i,:len(str)] =[int(s)-1 for s in str]
        
        inputs = {'the_input': x,
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                }
        outputs = {'ctc': np.zeros([batchsize])} 
        yield (inputs,outputs)



        

input = Input(shape=(img_h,None,1),name='the_input')

y_pred= densenet.dense_cnn(input,nclass, name='new_out')

basemodel = Model(inputs=input,outputs=y_pred)
basemodel.summary()


labels = Input(name='the_labels',shape=[maxlabellength],dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) 

model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)

adam = Adam()

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam,metrics=['accuracy'])


checkpoint = ModelCheckpoint(r'../../../keras_ocr_model/weights1-densent-{epoch:02d}.hdf5',
                             #save_weights_only=False,
                             save_best_only=True)
earlystop = EarlyStopping(patience=10)
tensorboard = TensorBoard(r'../../../keras_ocr_model/tflog-densent5',write_graph=True)

print('-----------beginfit--')
with tf.Session() as sess:
    cc1=gen3([r'../../../text_renderer-dev/train1/default/tmp_labels_id.txt'
              #r'/datasets/TextRecognitionDataGenerator/TextRecognitionDataGenerator/train1.txt',
              #r'/datasets/TextRecognitionDataGenerator/TextRecognitionDataGenerator/train2.txt'
              ],
              batchsize=batch_size,maxlabellength=maxlabellength,imagesize=(img_h,img_w))
    cc2=gen3([r'../../../text_renderer-dev/val1/default/tmp_labels_id.txt'],batchsize=batch_size,maxlabellength=maxlabellength,imagesize=(img_h,img_w))

# fine tune
from keras.models import load_model
from keras.models import Sequential
custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred}

model.load_weights(r'../../../keras_ocr_model/weights9-densent-10.hdf5', by_name=True)

res = model.fit_generator(cc1,
                    steps_per_epoch = 300000// batch_size,
                    epochs = 100,
                    validation_data =cc2 ,
                    validation_steps = 1000// batch_size,
                    callbacks =[earlystop,checkpoint,tensorboard],
                    verbose=1
                    )

