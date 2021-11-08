import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import os

class RNN(object):
    def __init__(self,inputshapex=1,inputshapey=35,output_dim=1,batch_size=128,verbose=True,netparamsdir='./',optimizer='rmsprop'):
        self.paramsidr = netparamsdir
        self.verbose =verbose
        self.optimizer = optimizer
        self.outputdim = output_dim
        self.rms_prop_optimizer =tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.00025) #tf.train.RMSPropOptimizer(learning_rate=0.00025)
       
        self.lstm = Sequential()        
        #self.lstm.add(LSTM(64,return_sequences=True,stateful=False,input_shape=(inputshapex,inputshapey),batch_size=batch_size)) 
        self.lstm.add(LSTM(64,return_sequences=True,stateful=False,input_shape=(inputshapex,inputshapey),activation='tanh'))
        self.lstm.add(LSTM(32,return_sequences=True,activation='tanh'))
        self.lstm.add(LSTM(32,activation='tanh'))
        self.lstm.add(Dense(self.outputdim))      
        self.lstm.add(Dense(1))#self.lstm.add(Dense(2,activation='softmax'))

    def compile(self,reload_weights_if_exists=True): #adam
        loaded = self.load_weights() 
        if(not loaded):
            loaded = self.load_states()       
        if(self.optimizer=='rmsprop'):
            self.lstm.compile(loss="mean_squared_error",optimizer=self.rms_prop_optimizer)
        else:
            self.lstm.compile(loss="mean_squared_error",optimizer='adam')

    def predict(self,x):
        return self.lstm.predict(x)

    def fit(self,x=None, y=None, batch_size=None, epochs=1, verbose=0, callbacks=None, validation_split=0., validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None):
        return self.lstm.fit(x,y,batch_size,epochs,verbose,callbacks,validation_split,validation_data,shuffle,class_weight,sample_weight,initial_epoch,steps_per_epoch,validation_steps)

    def save_weights(self):
        os.makedirs(self.paramsidr, exist_ok=True)
        paramdir = self.paramsidr + '/qlstm_weights.h5'
        self.lstm.save_weights(paramdir)
    def save_states(self):
        os.makedirs(self.paramsidr, exist_ok=True)
        paramdir = self.paramsidr + '/qlstm_states.h5'
        self.lstm.save(paramdir)
    def load_weights(self):
        loaded= False
        print("\nQLSTM TRY WEIGHTS LOADING!\n")
        paramdir = self.paramsidr + '/qlstm_weights.h5'
        if(os.path.exists(paramdir)):
            self.lstm.load_weights(paramdir)
            print("\nQLSTM WEIGHTS LOADED!\n")
            loaded = True
        return loaded

    def load_states(self):
        print("\nQLSTM TRY STATES LOADING!\n")
        loaded= False        
        paramdir = self.paramsidr + '/qlstm_states.h5'
        if(os.path.exists(paramdir)):
            self.lstm = keras.models.load_model(paramdir)
            print("\nQLSTM STATES LOADED!\n")
            loaded = True
        return loaded
    
