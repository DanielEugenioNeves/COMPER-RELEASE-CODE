import numpy as np
import keras
from keras.models import Sequential,Model,Input
from keras.layers import Dense, Dropout, Flatten,Activation,Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import keras.backend as K
import os
#sess = tf.Session()
#K.set_session(sess)
class ConvNetGscale(object):
    def __init__(self,netparamsdir='./'):
        self.paramsidr = netparamsdir
        self.convnet = {}
                
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00025,decay=0.95,momentum=0.95)        
    
    def create(self,output_dim=18):       

        self.convnet = tf.keras.Sequential([            
            tf.keras.layers.Conv2D(32,(8,8),strides=(4,4),data_format="channels_last",input_shape=(84,84,1),name="Conv1"),
            tf.keras.layers.Conv2D(64, (4,4),strides=(2,2),name="Conv2"),
            tf.keras.layers.Conv2D(64, (3,3),strides=(1,1),name="Conv3"),
            tf.keras.layers.Flatten(name="Flatten"),
            tf.keras.layers.Dense(512,name="Dense512"),
            tf.keras.layers.Dense(output_dim,activation=None,name="DenseOut")
        ])  

        #print(self.convnet.summary()) 
        #plot_model(self.convnet,to_file="convnet.png")       
    
    def compile_model(self):
        try:
            print("\nCONVNET TRY PARAMETER LOADING!\n")
            paramdir = self.paramsidr + '/dnn_weights.h5'
            if(os.path.exists(paramdir)):
                self.convnet.load_weights(paramdir)
                print("\nCONVNET PARAMETER LOADED!\n")
            
        except Exception as isnt:
            print(type(isnt))
            print(isnt)
            pass
        #self.convnet.compile(loss=keras.losses.mean_squared_error,optimizer=self.optimizer)
    
    def get_loss(self,states,qtargets):
        q = tf.reduce_sum(self.convnet(states),axis=1)
        #loss = tf.keras.losses.mean_squared_error(q,qtargets)
        loss = tf.losses.huber_loss(qtargets, q, reduction=tf.losses.Reduction.NONE)
        return tf.reduce_mean(loss)

    def grad(self,loss):            
       gradients = self.optimizer.get_gradients(loss, self.convnet.trainable_variables) # gradient tensors      
       return gradients

    def fit_model(self,states,qtargets):        
        s = tf.placeholder(dtype=tf.float32,shape=states.shape)
        qt= tf.placeholder(dtype=tf.float32,shape=qtargets.shape)

        loss_value = self.get_loss(s,qt)
        grads = self.grad(loss_value)       
        train_op =self.optimizer.apply_gradients(zip(grads,self.convnet.trainable_variables)) 

        init_op = tf.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init_op)        
        sess.run(train_op,feed_dict={s:states,qt:qtargets})
        sess.close()
        #with sess.as_default():            
           #train_op.run(feed_dict={s:states,qt:qtargets})
    
    def forward(self,state):        
        fop = self.convnet.predict_on_batch(state)
        return fop

    def save_weights(self):
        os.makedirs(self.paramsidr, exist_ok=True)
        paramdir = self.paramsidr + '/dnn_weights.h5'
        self.convnet.save_weights(paramdir)
    def save_states(self):
        os.makedirs(self.paramsidr, exist_ok=True)
        paramdir = self.paramsidr + '/dnn_states.h5'
        self.convnet.save(paramdir)