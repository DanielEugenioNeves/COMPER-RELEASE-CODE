
import numpy as np
from comper.memory.base.base_transitions_memory import BaseTransitionsMemory
from comper.memory.rtm.reduced_transitions_set import RTSet

class ReducedTransitionsMemory(BaseTransitionsMemory):
    def __init__(self,max_size,name,memory_dir='./'):        
        super().__init__(name,memory_dir='./') 
        self.rt_set ={}
        self.tmax = max_size
        
        
        self.__initialize()
    
    def __initialize(self):
        self.__init_rtset()
    
    def __init_rtset(self):
        self.rt_set = RTSet(max_sets=self.tmax)
    
    def __len__(self):
        return self.rt_set.len()     

    def __shape_transition(self,s_t_1,a_t_1,r_t,s_t,q_t,done):
        t = np.array(np.divide(s_t_1,255)).flatten()
        t = np.insert(t,len(t),a_t_1)        
        t = np.insert(t,len(t),r_t)
        t = np.insert(t,len(t),np.divide(s_t,255).flatten())
        t = np.insert(t,len(t),q_t)        
        t = np.insert(t,len(t),done)
        return np.array(t)

    def add_transition(self,s_t_1,a_t_1,r_t,s_t,q_t,done,prob=0):
        t = self.__shape_transition(s_t_1,a_t_1,r_t,s_t,q_t,done)      
        self.rt_set.add(t)

    def sample_transitions_batch(self,batch_size=32):
        samp = self.rt_set.random_samp(batch_size)        
        return samp
            
    def update_transition(self,key,t):
        self.rt_set.update(key,t)

        
        