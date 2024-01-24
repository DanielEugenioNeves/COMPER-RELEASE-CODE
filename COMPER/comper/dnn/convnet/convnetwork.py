import tensorflow as tf
import numpy as np
import os
from comper.config.transitions import FrameTransition as ft
from comper.config.transitions import FrameTransitionTypes as ft_types
from comper.config import parameters as param
from comper.config.exceptions import ExceptionRunType

class ConvNet(object):
    def __init__(self,netparamsdir='./',run_type=param.RunType.TRAIN):
        self.paramsidr = netparamsdir
        self.run_type = run_type
        self.convnet = {}
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00025)
    
    def create(self,input_shape,output_dim=18):
        if(ft.TYPE == ft_types.STAKED_FRAMES):
            self.__create_staked_frames_convnet(input_shape,output_dim)
        elif(ft.TYPE == ft_types.SINGLE_FRAMES):
            self.__create_single_frame_convnet(input_shape,output_dim)
    
    def __create_staked_frames_convnet(self,input_shape,output_dim):
        self.convnet = tf.keras.Sequential([            
            tf.keras.layers.Conv2D(32,(8,8),strides=(4,4),activation='relu',data_format="channels_last",input_shape=input_shape,use_bias=False,name="Conv1"),
            tf.keras.layers.Conv2D(64, (4,4),strides=(2,2),activation='relu',use_bias=False,name="Conv2"),
            tf.keras.layers.Conv2D(64, (3,3),strides=(1,1),activation='relu',use_bias=False,name="Conv3"),
            tf.keras.layers.Flatten(name="Flatten"),
            tf.keras.layers.Dense(512,activation='relu',use_bias=False,name="Dense512"),
            tf.keras.layers.Dense(output_dim,activation=None,name="DenseOut")
        ])
    def __create_single_frame_convnet(self,input_shape,output_dim):
        self.convnet = tf.keras.Sequential([            
            tf.keras.layers.Conv2D(32,(8,8),strides=(4,4),activation='relu',data_format="channels_last",input_shape=input_shape,name="Conv1"),
            tf.keras.layers.Conv2D(64, (4,4),strides=(2,2),activation='relu',use_bias=False,name="Conv2"),
            tf.keras.layers.Conv2D(64, (3,3),strides=(1,1),activation='relu',use_bias=False,name="Conv3"),
            tf.keras.layers.Flatten(name="Flatten"),
            tf.keras.layers.Dense(512,activation='relu',use_bias=False,name="Dense512"),
            tf.keras.layers.Dense(output_dim,activation=None,name="DenseOut")
        ])


    def compile_model(self):
        try:
            loaded = self.load_weights()
            if(not loaded):
                loaded = self.load_states()
            if(not loaded and self.run_type==param.RunType.TEST):
                raise ExceptionRunType(self.run_type,message="Weigths not found to run on test mode.")
        except Exception as isnt:
            print(type(isnt),isnt) 
            pass           
            
        #self.convnet.compile(loss=keras.losses.mean_squared_error,optimizer=self.optimizer)
    
    def get_loss(self,states,qtargets):
        q = tf.reduce_sum(self.convnet(states),axis=1)
        #loss = tf.keras.losses.mean_squared_error(q,qtargets)
        loss = tf.compat.v1.losses.huber_loss(qtargets, q, reduction=tf.compat.v1.losses.Reduction.NONE)
        return tf.reduce_mean(loss)

    def grad(self,loss):            
       gradients = self.optimizer.get_gradients(loss, self.convnet.trainable_variables) # gradient tensors      
       return gradients

    def fit_model(self,states,qtargets):        
        s = tf.compat.v1.placeholder(dtype=tf.float32,shape=states.shape)
        qt= tf.compat.v1.placeholder(dtype=tf.float32,shape=qtargets.shape)

        loss_value = self.get_loss(s,qt)
        grads = self.grad(loss_value)       
        train_op =self.optimizer.apply_gradients(zip(grads,self.convnet.trainable_variables)) 

        init_op = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init_op)        
        sess.run(train_op,feed_dict={s:states,qt:qtargets})
        sess.close()
        #with sess.as_default():            
           #train_op.run(feed_dict={s:states,qt:qtargets})
    
    def forward(self,state):
        #stateph =  tf.placeholder(dtype=tf.float32, shape=state.shape)
        fop = self.convnet.predict_on_batch(state)
        #pred = np.array
        #init_op = tf.global_variables_initializer()
        #sess.run(init_op)
        #with sess.as_default():            
           #pred = sess.run(fop,feed_dict={self.inputs:state})
        #return pred
        return fop

    def save_weights(self):
        os.makedirs(self.paramsidr, exist_ok=True)
        paramdir = self.paramsidr + '/dnn_weights.h5'
        self.convnet.save_weights(paramdir)
    def save_states(self):
        os.makedirs(self.paramsidr, exist_ok=True)
        paramdir = self.paramsidr + '/dnn_states.h5'
        self.convnet.save(paramdir)
            
    def load_weights(self):
        try:
            loaded= False
            print("\nCONVNET TRY WEIGHTS LOADING!\n")
            paramdir = self.paramsidr + '/dnn_weights.h5'
            print(paramdir)
            if(os.path.exists(paramdir)):
                print(paramdir)
                self.convnet.load_weights(paramdir)
                print("\nCONVNET WEIGHTS LOADED!\n")
                loaded = True
            return loaded
        except ValueError as isnt:
            print(type(isnt),isnt)
            raise isnt

    def load_states(self):
        print("\nCONVNET TRY STATES LOADING!\n")
        loaded= False        
        paramdir = self.paramsidr + '/dnn_states.h5'
        if(os.path.exists(paramdir)):
            self.convnet = tf.keras.models.load_model(paramdir)
            print("\nCONVNET STATES LOADED!\n")
            loaded = True
        return loaded


    



              




              
