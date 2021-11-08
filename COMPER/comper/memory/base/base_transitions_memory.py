import pickle
import os
from comper.config.transitions import FrameTransition as FT

class BaseTransitionsMemory(object):
    def __init__(self,memory_name,memory_dir='./'):
        self.name = memory_name
        self.memory_dir = memory_dir        
        self.tlen = FT.T_LENGTH
        self.st_w = FT.ST_W
        self.st_h = FT.ST_H
        self.st_l = FT.ST_L
        self.t_n_index = FT.T_N_IDX
        self.t_idx_st_1 = FT.T_IDX_ST_1
        self.t_idx_a = FT.T_IDX_A
        self.t_idx_r = FT.T_IDX_R
        self.t_idx_st = FT.T_IDX_ST
        self.t_idx_q = FT.T_IDX_Q
        self.t_idx_done = FT.T_IDX_DONE

        self.memory={}

    def save_memory_content(self):
        try:
            path =  self.memory_dir+'/'+self.name+'.pkl'
            os.makedirs(self.memory_dir, exist_ok=True)
            output = open(path,'wb')
            pickle.dump(self.memory,output)
        except Exception as e:
            print(type(e),e)          
            pass

    def load_memory_content(self):
        try:
            path =  self.memory_dir+'/'+self.name+'.pkl'
            if(os.path.exists(path) and os.path.getsize(path)>0):                
                pkl_file = open(path,'rb')
                self.memory = pickle.load(pkl_file)
                pkl_file.close()               
        except Exception as e:
            print(type(e),e)            
            pass