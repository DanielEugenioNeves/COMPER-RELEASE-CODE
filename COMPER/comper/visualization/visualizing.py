from datetime import datetime
from comper.visualization.salience_map import SalienceMap
from comper.visualization.clustering_pca import CluesteringPCA
from comper.visualization.clustering_tsne import ClusteringTSNE
from collections import defaultdict
import numpy as np

class Visualizing(object):
    def __init__(self,rom_name):
        self.rom_name = rom_name        
        self.salience_map_enabled = False
        self.salience_map_dir = ""
        self.tsne_clustering_enabled = False
        self.tsne_clustering_dir = ""
        self.pac_frames_clustering_enabled = False
        self.pca_frames_clustering_dir = ""
        self.frames=[]
        self.actions =[]
        self.q_values=[]
        self.rewards = []
        self.game_over_signals =[]
        self.real_frames=[]    
        self.history_count=0
        self.salience_map_count=0
        self.pca_clustering_count = 0
        self.tsne_clustering_count = 0
        self.convnet=None
    

    def enable_salience_map(self,salience_map_dir):
        self.salience_map_enabled = True
        self.salience_map_dir = salience_map_dir
    def enable_tsne_clusterint(self,tsne_clustering_dir):
        self.tsne_clustering_enabled = True
        self.tsne_clustering_dir = tsne_clustering_dir
    def enable_pca_frames_clustering(self,pca_frames_clustering_dir):
        self.pac_frames_clustering_enabled = True
        self.pca_frames_clustering_dir = pca_frames_clustering_dir 

    def set_convnet(self,convnet):
        self.convnet = convnet
    
    def add_history(self,st_1,action,reward,q_value,game_over_signal,real_frame):
        if(self.__can_add_history()):
            self.frames.append(st_1)
            self.actions.append(action)
            self.rewards.append(reward)
            self.q_values.append(q_value)
            self.game_over_signals.append(game_over_signal)
            self.real_frames.append(real_frame)
            self.history_count+=1    

    def run_visualization(self):
        if(self.__can_run()):
            if self.salience_map_enabled :
                self.__apply_salience_map()
            if self.tsne_clustering_enabled:
                self.__tsne_activations_clustering()
            if self.pac_frames_clustering_enabled:
                self.__pca_frames_clustering()

    def __apply_salience_map(self):
        salience_map = SalienceMap()
        for i in range(self.history_count):
            now = datetime.now()        
            dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
            image_name = self.rom_name+"_"+str(i)+dt_string+".png"
            salience_map.apply_salience_map(self.frames[i],self.convnet,self.salience_map_dir,image_name,True,self.real_frames[i])    

    def apply_salience_map(self,frames,real_frames):
        self.history_count+=1
        now = datetime.now()        
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        image_name = self.rom_name+"_"+str(self.history_count)+dt_string+".png"
        salience_map = SalienceMap()
        salience_map.apply_salience_map(frame=frames,convnet=self.convnet,save_img_path=self.salience_map_dir,image_name=image_name,smooth=True,real_frame=real_frames)

    def __tsne_activations_clustering(self):
        now = datetime.now()        
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        image_name = self.rom_name+"_"+str(self.history_count)+dt_string+".png"
        tsne = ClusteringTSNE()
        tsne.clustering(self.frames,self.convnet,"Dense512",self.q_values,self.tsne_clustering_dir,image_name)

    def __pca_frames_clustering(self):
        features =[]
        game_over = 0
        for i in range(self.history_count):
            features_row = np.array(self.frames[i])          
            features_row = features_row.flatten()            
            features_row = np.insert(features_row,len(features_row),float(self.actions[i]))
            features_row = np.insert(features_row,len(features_row),float(self.rewards[i]))
            features_row = np.insert(features_row,len(features_row),float(self.q_values[i]))
            game_over =  1 if (self.game_over_signals[i]) else 0            
            features_row = np.insert(features_row,len(features_row),float(game_over))
            features.append(features_row)
        now = datetime.now()        
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        image_name = self.rom_name+"_"+str(self.history_count)+dt_string+".png"
        pca = CluesteringPCA(np.array(features))
        pca.clustering(self.pca_frames_clustering_dir,image_name)
        
    def __can_run(self):
        can_run = True
        if(self.history_count==0):
            print("WARNING: to run visualization processes first set the history of frames")
            can_run = False
        if(self.convnet==None):
            print("WARNING: to run visualization processes first set the CNN")
            can_run = False
        return can_run
    def __can_add_history(self):
        can_add = (self.enable_salience_map or self.enable_tsne_clusterint or self.enable_pca_frames_clustering)
        if(not can_add):
            print("WARNING:  to add history first enable one or more visualization method")
        return can_add