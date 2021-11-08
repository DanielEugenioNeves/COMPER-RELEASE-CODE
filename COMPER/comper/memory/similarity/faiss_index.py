import numpy as np
import faiss

class FaissIndex(object):
    def __init__(self,max_transitions,transition_length):
        self.maxt = max_transitions
        self.tlen = transition_length                
        self.index = faiss.IndexFlatL2(self.tlen)        
        self.memory =[]
        self.init=False

    def initialized(self):
        return self.init
    
    def __check_maxt(self):
        if(len(self.memory)>self.maxt):
            self.memory = self.memory[:-1, :]

    def __norm_and_tshape(self,t):               
        t = t.reshape(1,self.tlen).astype('float32')
        return t

    def update_index(self):
        self.index.reset()
        self.index.add(self.memory)

    def add_transition(self,t):
        t = self.__norm_and_tshape(t)
        if not self.init:            
            self.memory = t            
            self.init=True                  
        else:
            self.__check_maxt()            
            self.memory= np.concatenate((self.memory,t),axis=0)
        self.update_index()
        return self.index.ntotal

    def reduce(self,n):        
        self.memory = self.memory[:-n, :]
    
    def sanity_check(self):
        k = 1                          # we want to see 4 nearest neighbors
        D, I =self.index.search(self.memory[:1], k) # sanity check
        print("Faiss Sanity")
        print(I)
        print(D)

    def get_sim_transition(self,t,k=1):
        distance, index = [[-1.0]],[[-1.0]]
        t = self.__norm_and_tshape(t)       
        if self.init:
            distance, index=self.index.search(t, k)
        return distance,index
               
        
    
    

    

