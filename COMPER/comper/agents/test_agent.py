import numpy as np
from datetime import datetime
from comper.agents.base_agent import BaseTestAgent
from comper.dnn.convnet.convnetwork import ConvNet
from comper.config.transitions import FrameTransition as ft
from comper.config.transitions import FrameTransitionTypes as f_type
from comper.config import parameters as param

class TestAgent(BaseTestAgent):
    def __init__(self,rom_name,rom_file_path,log_dir,nets_param_dir,display_screen=param.DisplayScreen.OFF,framesmode=param.FrameMode.SINGLE):
        super().__init__(rom_name=rom_name,rom_file_path=rom_file_path,log_dir=log_dir,
                        nets_param_dir=nets_param_dir,display_screen=display_screen,framesmode=framesmode)

        self.__initialize()
        
    def __initialize(self):
        print("Init test agent with "+self.framesmode+"frames")
        super().initialize()    
    
    def __get_env_state(self):
        st_1 = self.env.getScreenGrayscale()
        st_1 = self.statePreprocessor.gray_preprocess_obs_ram(st_1)
        st_1 = np.array(np.divide(st_1,255))
        return st_1
        
    def __agente_step(self):
        st_1 = self.__get_env_state()
        q_values = self.q.forward(st_1)        
        action = np.argmax(q_values, axis=1).data[0]
        reward = self.env.act(action)
        return reward

    def __compute_trial(self,trial):
        n_frames = self.env.getEpisodeFrameNumber()
        self.total_frames+=n_frames
        self.sumTrialsRewards+=self.trialReward
        self.trial_count+= trial        
        self.LogTestData(trial,self.trialReward,n_frames,self.sumTrialsRewards,self.total_frames)

    def RunTest(self,nTrials):
        self.n_trials = nTrials
        print("Test Agent Running Test!")        
        for trial in range(0,self.n_trials):
            self.trialReward =0
            self.reset_game()            
            while(not self.env.game_over()):
                r = self.__agente_step()
                self.trialReward+=r
            self.__compute_trial(trial)

    


       
