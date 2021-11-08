import sys
import numpy as np
from collections import deque
from numpy.random import RandomState
from collections import deque
from pandas import DataFrame
import pandas
from datetime import datetime
from comper.config import exceptions as ex
from comper.agents.base_agent import BaseTrainAgent
from comper.memory.tm.transitions_memory import TransitionsMemory as TM
from comper.memory.rtm.reduced_transitions_memory import ReducedTransitionsMemory as RTM
from comper.dnn.convnet.convnetwork import ConvNet
from comper.qrnn.q_lstm_gscale import QLSTMGSCALE
from comper.utils.class_lib import Epsilon,Epsilon2
from comper.config.transitions import FrameTransition as ft
from comper.config.transitions import FrameTransitionTypes as f_type
from comper.config import parameters as param

class Agent(BaseTrainAgent):
    def __init__(self,rom_name,rom_file_path,maxtotalframes,frames_ep_decay,train_frequency,update_target_frequency,learning_start_iter,log_frequency,log_dir,
                nets_param_dir,memory_dir,save_states_frq,persist_memories,save_networks_weigths=True,save_networks_states=0,
                display_screen=param.DisplayScreen.OFF,framesmode=param.FrameMode.SINGLE):
                        
        super().__init__(rom_name,rom_file_path,maxtotalframes,frames_ep_decay,train_frequency,update_target_frequency,learning_start_iter,log_frequency,log_dir,
                        nets_param_dir,memory_dir,save_states_frq,persist_memories,save_networks_weigths,save_networks_states,display_screen,framesmode)
        
        self.q_input_shape = (ft.ST_W, ft.ST_H, ft.ST_L)
        self.q_target_input_shape = ft.T_LENGTH -2
        self.transitonsBatchSize = 32
        self.discountFactor = 0.99
        self.q = {}
        self.qt={}
        self.tm={}
        self.rtm = {}       
        self.__initialize()

    def __initialize(self):
        print("Init train agent with "+self.framesmode+"frames")
        self.__define_epsilon()        
        self.__schedule_epsilon()
        self.__config_environment()
        self.__config_memories()        
        self.__create_q_network()
        self.__create_qt_network()        
    
    def __define_epsilon(self):        
        self.epsilonInitial = 1.0
        self.epsilonFinal = 0.001
        self.epsilonFraction = 0.0
        if(ft.TYPE==f_type.SINGLE_FRAMES and self.frames_ep_decay==90000):
            self.epsilonFraction = 0.18016 
        elif(ft.TYPE==f_type.SINGLE_FRAMES and self.frames_ep_decay==100000):
            self.epsilonFraction = 0.20099
        elif(ft.TYPE==f_type.STAKED_FRAMES and self.frames_ep_decay==90000):
            self.epsilonFraction = 19.95339547

    def __schedule_epsilon(self):       
        if(ft.TYPE==f_type.SINGLE_FRAMES):
            self.epsilon = Epsilon(schedule_timesteps=int(self.epsilonFraction * self.maxtotalframes),initial_p=self.epsilonInitial,final_p=self.epsilonFinal)
        elif(ft.TYPE==f_type.STAKED_FRAMES):
            self.epsilon = Epsilon2(init_exploration_rate=self.epsilonInitial,final_exploration_rate=self.epsilonFinal,
                                final_exploration_frame=self.frames_ep_decay,fraction=self.epsilonFraction)


    def __config_memories(self):               
        self.tm = TM(max_size=100000,name="tm", memory_dir=self.memory_dir)
        self.rtm  = RTM(max_size=100000,name="rtm",memory_dir=self.memory_dir)
            
    def __config_environment(self):
        super().config_environment()

    def __create_q_network(self):        
        self.q = ConvNet(netparamsdir=self.nets_param_dir)
        self.q.create(input_shape = self.q_input_shape,output_dim=self.nActions)
        self.q.compile_model()
    
    def __create_qt_network(self):                   
        self.qt = QLSTMGSCALE(transitions_memory=self.tm,reduced_transitions_memory=self.rtm,
                              inputshapex=1,inputshapey=self.q_target_input_shape,outputdim=self.nActions,
                              verbose=False,transition_batch_size=1000,netparamsdir=self.nets_param_dir,
                              target_optimizer=self.target_optimizer,log_dir=self.logDir)

    def __select_epsilon_greedy(self, state, epsilon):        
        rnd = self.randon.rand()        
        q_values = self.q.forward(state)        
        action=0
        max_q_value =0        
        if rnd > epsilon:                                 
            action = np.argmax(q_values, axis=1).data[0]            
            max_q_value = float(q_values[0][action])
        else:            
            action = self.randon.randint(0,self.nActions)            
            max_q_value = float(q_values[0][action])
        
        return action,max_q_value    

    def _get_transition_components(self,transitions):
        st_1 = transitions[:,:ft.T_IDX_ST_1[1]]
        a = transitions[:,ft.T_IDX_A]
        r = transitions[:,ft.T_IDX_R] # get rewards
        st = transitions[:,ft.T_IDX_ST[0]:ft.T_IDX_ST[1]]
        q = transitions[:,ft.T_IDX_Q]# To ilustrate, but we do not need this here.
        done = transitions[:,ft.T_IDX_DONE] # get done signals
        return st_1,a,r,st,q,done
    
    def _get_discounted_reward(self,r,target_predicted_q,done):
        _q = r + (1 - done) * self.discountFactor * target_predicted_q
        _q=np.float32(_q)
        return _q

    def __comper_loss(self,transitions):            
        try:
            st_1,a,r,st,q,done = self._get_transition_components(transitions)
            transitions = transitions[:,:-2]            
            target_predicted = self.qt.predict(transitions)            
            target_predicted_q =target_predicted[:,-1]

            y = self._get_discounted_reward(r,target_predicted_q,done)
            
            reshape_st_1 = st_1.reshape([-1,self.q_input_shape[0],self.q_input_shape[1],self.q_input_shape[2]])           
                
            self.q.fit_model(states =reshape_st_1 ,qtargets =np.array(y.data))            
            q = self.q.forward(reshape_st_1)

            _q = np.empty(q.shape[0])        
            for i in range(0,len(q)):
                _q[i] = q[i,int(a[i])] 

            for i in range(len(st_1)):                 
                self.tm.add_transition(st_1[i],a[i],r[i],st[i],_q[i],float(done[i]))

            loss = (np.square(y- _q)).mean(axis=None)     
            return loss

        except Exception as e:
            print(type(e),e)           
            raise(e)

    def _get_sample_transitions(self):
        transitions=[]
        if(self.rtm. __len__()>0):
            transitions = self.rtm.sample_transitions_batch(self.transitonsBatchSize)
        else:
            transitions = self.tm.sample_transitions_batch(self.transitonsBatchSize)
        return transitions

    def __train_q(self,transitions):
        loss = self.__comper_loss(transitions)
        return loss

    def __train_qt(self):
        self.qt.train_q_prediction()
    
    def __check_train_qt(self,itr):
        if (itr % self.trainQTFreqquency == 0 and itr > self.learningStartIter):               
                self.__train_qt()  

    def __check_train_q(self,itr,tde_list):                
        if (itr % self.trainQFrequency == 0 and itr >self.learningStartIter):
            transitions=self._get_sample_transitions()                   
            tde = self.__train_q(transitions)
            tde_list.append(tde)

    def __get_env_state(self):
        st = self.env.getScreenGrayscale()
        st = self.statePreprocessor.gray_preprocess_obs_ram(st)            
        return st

    def __get_env_stacked_states(self, action):
        i = 0
        immediate_reward = 0
        raw_states = []
        while i < 4:            
            state = self.__get_env_state()
            immediate_reward += self.env.act(action)
            raw_states.append(state)
            i += 1        
        return immediate_reward, np.dstack(raw_states)

    def __make_staked_transition(self,st_1,e):
        _st_1 = np.array(np.divide(st_1,255)).reshape([-1,ft.ST_W, ft.ST_H,ft.ST_L])
        a, q = self.__select_epsilon_greedy(_st_1,e)
        r,st = self.__get_env_stacked_states(a)        
        return st_1,a,r,st,q
    
    def _make_transition(self,st_1,e):        
        _st_1 = np.array(np.divide(st_1,255))        
        a, q = self.__select_epsilon_greedy(_st_1,e)
        r = self.env.act(a)
        st = self.__get_env_state()
        return st_1,a,r,st,q

    def _agent_step(self,itr,st_1,done):        
        if ft.TYPE == f_type.SINGLE_FRAMES:
            e = self.epsilon.value(itr)
            st_1,a,r,st,q =self._make_transition(st_1,e)
        elif ft.TYPE == f_type.STAKED_FRAMES:
            e = self.epsilon.value()
            st_1,a,r,st,q =self.__make_staked_transition(st_1,e)
        self.tm.add_transition(st_1,a,r,st,q,float(done))            
        return r,e,st

    def _firt_env_state(self):
        state = []
        self.reset_game()
        if ft.TYPE == f_type.SINGLE_FRAMES:
            state = self.__get_env_state()
        elif ft.TYPE == f_type.STAKED_FRAMES:
            _,state = self.__get_env_stacked_states(0)
        return state
        
    def train_agent(self):
        rewards =deque([])
        episode_rewards = deque([])
        tde_list = deque([])         
        current_episode_rewards=0
        total_current_episode_reward=0        
        n_episodes = 0
        log_itr=-1
        done =False
        episode_frames_number=0
        total_frames_number =0
        itr = 1
        e=0       
        run = True
        st_1 = self._firt_env_state()
        while(run):
            itr+=1
            r,e,st_1 = self._agent_step(itr,st_1,done)
            rewards.append(r)
            current_episode_rewards+=r
            total_current_episode_reward = current_episode_rewards
            done = self.env.game_over()    

            if (not done):                
                if((itr+1) % self.logFrequency == 0 or total_frames_number == 90000):            
                    log_itr+=1
                    self.LogAgentTrainData(log_itr,itr,e,n_episodes,rewards,total_current_episode_reward,episode_rewards,
                                            tde_list,done,episode_frames_number,total_frames_number)
            else:
                log_itr+=1                
                episode_rewards.append(np.sum(rewards))               
                self.LogAgentTrainData(log_itr,itr,e,n_episodes,rewards,total_current_episode_reward,episode_rewards,
                                        tde_list,done,episode_frames_number,total_frames_number)
                self.reset_game()
                current_episode_rewards=0                
                n_episodes+=1
                run = (self.env.getFrameNumber()<=self.maxtotalframes)

            total_frames_number =self.env.getFrameNumber()    
            episode_frames_number = self.env.getEpisodeFrameNumber()
            
            self.__check_train_qt(itr)
            self.__check_train_q(itr,tde_list)
            self.__save_states(itr)                
    
    def __save_states(self,itr):
        if((itr+1) % self.save_states_freq == 0 ):
            if(self.save_networks_weigths):
                self.q.save_weights()
                self.qt.save_weights()                         

            if(self.save_networks_states==1):
                self.q.save_states()                   
                #self.qt.save_states()

            if(self.persist_memories):
                self.tm.save_memory_content()
                self.rtm.save_memory_content()         
        
    def LogAgentTrainData(self,log_itr_count,itr,epsilon,n_episodes,rewards,total_current_episode_reward,episode_rewards,tde_list,done,episode_frames_number,total_frames_number):  
        now = datetime.now()        
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        log_data_dict =[
            ('Count', log_itr_count),
            ('Rom',self.rom_name),
            ('Time',dt_string),
            ('Itr', itr),
            ('EpFramesNumber', episode_frames_number),
            ('TMCount',self.tm.__len__()),
            ('RTMCount',self.rtm.__len__()),
            ('Eps', epsilon),
            ('Episodes', n_episodes),
            ('EndEp', done),
            ('CurrEpRewardSum', total_current_episode_reward),
            ('EpRewardsSum', np.sum(rewards)),
            ('AvgEpReturn', np.mean(episode_rewards)),
            ('TDE', np.mean(tde_list)),
            ('TotalFramesNumber', total_frames_number)]
        self.log(log_data_dict)