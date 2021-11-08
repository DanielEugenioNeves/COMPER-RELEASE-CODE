import numpy as np
import random
from comper.config.transitions import FrameTransition as ft

class RTSet(object):

    def __init__(self,max_sets):      
        self.msets = max_sets 
        self.tlen = ft.T_LENGTH
        self.q_idx = ft.T_IDX_Q       
        self.init = False
        self.sets =[]      
        self.sets_count =0       
    
    def len(self):
        return self.sets_count

    def __check_size(self,size):
        if(size>self.sets_count-1):
                size = self.sets_count-1
        return size
    
    def add(self,t):
        t = t.reshape(1,t.shape[0])    
        if not self.init:
            self.sets = t            
            self.init=True
            self.sets_count+=1
        else: 
            if self.sets_count >= self.msets:
                self.sets = self.sets[:-1,:]
                self.sets_count-=1
            self.sets= np.concatenate((self.sets,t),axis=0)
            self.sets_count+=1

    def random_samp(self,size):
        size = self.__check_size(size)
        samp=[]        
        try:
            idxes = sorted(random.sample(range(0,self.len()-1),size))            
            samp = self.sets[idxes,:]           
            return samp          
        except Exception as e:            
            raise(e)
    
    def update(self,key,t):
        try:
            if(key>self.sets_count):
                self.add(t)
            else:
                self.sets[key,self.q_idx]=t[self.q_idx]
        except Exception as e:            
            raise(e) 
            
    
                 

