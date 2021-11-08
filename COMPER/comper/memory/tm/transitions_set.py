import numpy as np
from collections import defaultdict,deque
import random
from comper.config.transitions import FrameTransition as ft

class TSet(object):
    def __init__(self,max_sets):      
        self.msets = max_sets
        self.tlen = ft.T_LENGTH        
        self.sets =defaultdict(list)      
        self.sets_count =0       
    
    def len(self):
        return self.sets_count

    def __check_size(self,size):
        if(size>self.sets_count):
                size = self.sets_count
        return size
    
    def add(self,t,key):        
        if key in self.sets:
            self.sets[key]=t
        else:
            if(self.sets_count>=self.msets):
                 self.sets.pop(list(self.sets.keys())[0])
                 self.sets_count-=1
            self.sets[key]=t                          
            self.sets_count+=1
    
    def batch(self,size,remove=True):
        size = self.__check_size(size)
        samp=defaultdict(list)
        try:
            keys = np.array(list(self.sets.keys()))
            keys = keys[:size]
            for k in keys:
                t = self.sets.pop(k) if remove else self.sets[k]
                t=t.reshape(1,self.tlen)
                samp[k]=t
                self.sets_count-=1            
            return samp          
        except Exception as e:            
            raise(e)

    def batch_tolist(self,size):
        size = self.__check_size(size)
        transitions=[]
        try:                                  
            keys = np.array(list(self.sets.keys()))
            keys = keys[:size]
            for k in keys:
                t = np.array(self.sets[k])                
                t = t.reshape(1,self.tlen)
                if(len(transitions)==0):
                    transitions = t
                else:
                    transitions = np.concatenate((transitions,t),axis=0)
            return np.array(transitions)
        except Exception as e:            
            raise(e)
            
    def random_samp(self,size):
        size = self.__check_size(size)
        samp=defaultdict(list)
        try:
            idxes = sorted(random.sample(range(0,self.len()-1),size))            
            keys = np.array(list(self.sets.keys()))
            keys = keys[idxes]
            for k in keys:
                samp[k]=self.sets[k]            
            return samp          
        except Exception as e:            
            raise(e)
