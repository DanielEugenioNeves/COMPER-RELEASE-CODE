from datetime import datetime
import numpy as np
from comper.dnn.convnet.convnetwork import ConvNet
from comper.visualization.visualizing import Visualizing
from comper.agents.base_agent import BaseTestAgent
from comper.config.transitions import FrameTransition as ft
from comper.config.transitions import FrameTransitionTypes as f_type
from comper.config import parameters as param



class TestVisualizingAgent(BaseTestAgent):
    def __init__(self,rom_name,rom_file_path,log_dir,nets_param_dir,display_screen=param.DisplayScreen.OFF,framesmode=param.FrameMode.SINGLE,
                salience_map=False,salience_map_dir="./",tsne=False,tsne_clustering_dir="./",pca_frames_clustering=False,pca_clustering_frames_dir="./" ):
        super().__init__(rom_name=rom_name,rom_file_path=rom_file_path,log_dir=log_dir,
                        nets_param_dir=nets_param_dir,display_screen=display_screen,framesmode=framesmode)        
        
        
        self.s_map =salience_map
        self.s_map_dir = "./visualizing/salience/"+ salience_map_dir
        self.tsne = tsne
        self.tsne_dir = "./visualizing/tsne/"+ tsne_clustering_dir
        self.pca_frames = pca_frames_clustering
        self.pca_frames_dir = "./visualizing/pca/"+ pca_clustering_frames_dir

        self.visualizing = {}
        self.visualizing_enabled = False
        self.history=[]

        self.__initialize()      
             
        
    def __initialize(self):
        print("Start testing and visualizing agent with "+self.framesmode+"frames")
        super().initialize()
        self.__config_visualizing()
    
    def __config_visualizing(self):
        self.visualizing_enabled = self.s_map or self.tsne or self.pca_frames
        if(self.visualizing_enabled):
            self.visualizing = Visualizing(self.rom_name)
            self.visualizing.set_convnet(self.q.convnet)
            self.__config_salience_map()
            self.__config_tsne_clustering()
            self.__config_pca_frames_clustering()
    
    def __config_salience_map(self):
        if(self.s_map):
            print("Salience Map Enabled")
            self.visualizing.enable_salience_map(self.s_map_dir)
    
    def __config_tsne_clustering(self):
        if(self.tsne):
            print("Tsne Clustering Enabled")
            self.visualizing.enable_tsne_clusterint(self.tsne_dir)
    
    def __config_pca_frames_clustering(self):
        if(self.pca_frames):
            print("PCA Clustering Enabled")
            self.visualizing.enable_pca_frames_clustering(self.pca_frames_dir)

    def __get_env_state(self):
        st_1 = self.env.getScreenGrayscale()       
        return st_1
    
    def __preprocess_env_state(self,st_1):
        st_1 = self.statePreprocessor.gray_preprocess_obs_ram(st_1)
        st_1 = np.array(np.divide(st_1,255))
        return st_1

    def __agente_step(self):
        raw_st_1 = self.__get_env_state()
        st_1 = self.__preprocess_env_state(raw_st_1)
        q_values = self.q.forward(st_1)        
        action = np.argmax(q_values, axis=1).data[0]
        reward = self.env.act(action)        
        return raw_st_1,st_1,action,reward,q_values

    def __compute_trial(self,trial):
        n_frames = self.env.getEpisodeFrameNumber()
        self.total_frames+=n_frames
        self.sumTrialsRewards+=self.trialReward
        self.trial_count+= trial        
        self.LogTestData(trial,self.trialReward,n_frames,self.sumTrialsRewards,self.total_frames)

    def __add_to_history(self,st_1,action,reward,q_values,game_over,raw_st_1):
        if(self.visualizing_enabled):
            self.visualizing.add_history(st_1,action,reward,q_values[0][action],game_over,raw_st_1)
    
    def __run_visulization(self):
        if(self.visualizing_enabled):
            self.visualizing.run_visualization()

    def RunTest(self,nTrials):
        self.n_trials = nTrials
        print("Testing and Visulizing Agent!")        
        for trial in range(0,self.n_trials):
            self.trialReward =0
            self.reset_game()
            game_over = False            
            while(not self.env.game_over()):
                raw_st_1,st_1,a,r,q = self.__agente_step()
                self.trialReward+=r
                self.__add_to_history(st_1,a,r,q,game_over,raw_st_1)
                game_over = self.env.game_over()
            self.__compute_trial(trial)
        self.__run_visulization()
     

    