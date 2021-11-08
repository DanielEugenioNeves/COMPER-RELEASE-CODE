from datetime import datetime
from comper.data import logger
from ale.ale_python_interface import ALEInterface
from comper.dnn.convnet.preprocessing import Preprocessor
from comper.dnn.convnet.convnetwork import ConvNet
from comper.config import exceptions as ex
from comper.config import parameters as param
from numpy.random import RandomState
from comper.config.transitions import FrameTransition as ft
from comper.config.transitions import FrameTransitionTypes as f_type

class BaseAgent(object):
    def __init__(self,rom_name,rom_file_path,log_dir,nets_param_dir,framesmode,display_screen,run_type=param.RunType.NO_SET):
        self.run_type =run_type
        self.rom_name = rom_name
        self.env = ALEInterface()
        self.statePreprocessor={}
        self.logDir = ""
        self.nets_param_dir= ""
        self.nActions=0
        self.framesmode = framesmode
        self.romPath= rom_file_path        
        self.displayScreen = display_screen
        self.__validate_base_parameters(framesmode,display_screen)

        self.__validate_run_type()
        self.__set_frames_mode()
        self.__set_nets_param_dir(nets_param_dir)
        self.__set_logs_dir(log_dir)
        self.__init_state_preprocessor() 
        self.__init_logger()
    
    def __validate_run_type(self):
        if(self.run_type==param.RunType.NO_SET):
            raise ex.ExceptionRunType(self.run_type)


    def __validate_base_parameters(self,framesmode,display_screen):
        if framesmode != param.FrameMode.SINGLE and framesmode !=param.FrameMode.STAKED:
            raise ex.ExceptionFrameMode(self.framesmode)
        if display_screen != param.DisplayScreen.ON and display_screen !=param.DisplayScreen.OFF:
            raise ex.ExceptionDisplayScreen(display_screen)          
    
    def __set_frames_mode(self):                
        if self.framesmode == param.FrameMode.SINGLE:
            ft.turn_single_frames()
        elif self.framesmode == param.FrameMode.STAKED:
            ft.turn_staked_frames()
    
    def __set_nets_param_dir(self,nets_param_dir=""):       
        if self.framesmode == param.FrameMode.SINGLE:
            self.nets_param_dir =  "./netparams/single_frames/"+nets_param_dir
        elif self.framesmode == param.FrameMode.STAKED:
            self.nets_param_dir =  "./netparams/staked_frames/"+nets_param_dir

    def __set_logs_dir(self,log_dir):
        base_dir = "./log_" 
        if self.run_type==param.RunType.TRAIN:
            base_dir = base_dir+"train/"        
        elif self.run_type==param.RunType.TEST:
            base_dir = base_dir+"test/"        
        if self.framesmode == param.FrameMode.SINGLE:
             self.logDir = base_dir+"single_frame/"+log_dir
        elif self.framesmode == param.FrameMode.STAKED:
            self.logDir =  base_dir+"staked_frames/"+log_dir

    def __init_state_preprocessor(self):
        self.statePreprocessor = Preprocessor()

    def __init_logger(self):
        logger.session(self.logDir).__enter__()
    
    def log(self,log_data_dict):
        for k, v in log_data_dict:
            logger.logkv(k, v)
        logger.dumpkvs()

    def reset_game(self):
        self.env.reset_game()

    def config_environment(self,color_averaging=True,repeat_action_probability=0.25,frame_skip=5):
        self.env.setInt(b'random_seed', 123)
        self.env.setBool(b'display_screen', self.displayScreen)
        self.env.setBool(b'color_averaging',color_averaging)
        self.env.setFloat(b'repeat_action_probability',repeat_action_probability)
        self.env.setInt(b'frame_skip',frame_skip)
        self.env.loadROM(self.romPath)        
        self.actions = self.env.getLegalActionSet()       
        self.nActions = len(self.actions)
    
class BaseTrainAgent(BaseAgent):
    def __init__(self,rom_name,rom_file_path,maxtotalframes,frames_ep_decay,train_frequency,update_target_frequency,learning_start_iter,log_frequency,
                 log_dir,nets_param_dir,memory_dir,save_states_frq,persist_memories,save_networks_weigths=True,
                 save_networks_states=0,display_screen=False,framesmode=param.FrameMode.SINGLE):
        
        super().__init__(rom_name,rom_file_path,log_dir,nets_param_dir,framesmode,display_screen,param.RunType.TRAIN)       
        
        self.maxtotalframes = maxtotalframes
        self.frames_ep_decay = frames_ep_decay
        self.trainQFrequency = train_frequency
        self.trainQTFreqquency = update_target_frequency
        self.learningStartIter = learning_start_iter
        self.logFrequency = log_frequency
        self.save_states_freq = save_states_frq
        self.persist_memories = persist_memories
        self.save_networks_weigths = save_networks_weigths
        self.save_networks_states = save_networks_states
        self.randon = RandomState(123)
        self.target_optimizer = "rmsprop"
        self.memory_dir=""

        self.__set_memory_dir(memory_dir)
        self.__validate_train_parameters()

    def __set_memory_dir(self,memory_dir):
        self.memory_dir="./transitions_memory/"+memory_dir
        
    def __validate_train_parameters(self):
        if self.persist_memories != param.PersistMemory.YES and self.persist_memories != param.PersistMemory.NO:
            raise ex.ExceptionPersistMemory(self.persist_memories)
           
class BaseTestAgent(BaseAgent):
    def __init__(self, rom_name,rom_file_path,log_dir,nets_param_dir,display_screen=param.DisplayScreen.OFF,framesmode=param.FrameMode.SINGLE):
        super().__init__(rom_name, rom_file_path, log_dir, nets_param_dir, framesmode, display_screen,param.RunType.TEST)        
        
        self.q_input_shape = (ft.ST_W, ft.ST_H, ft.ST_L)
        self.q = {}
        self.trialReward =0
        self.sumTrialsRewards=0
        self.n_trials=1
        self.trial_count=0
        self.total_frames=0

    def initialize(self):
        print("Init test agent with "+self.framesmode+"frames")
        self.__config_environment()
        self.__create_q_network()

    def __config_environment(self):        
        super().config_environment(repeat_action_probability=0,frame_skip=5)
    
    def __create_q_network(self):
        self.q = ConvNet(netparamsdir=self.nets_param_dir,run_type=param.RunType.TEST)
        self.q.create(input_shape = self.q_input_shape,output_dim=self.nActions)
        self.q.compile_model()

    def LogTestData(self,current_trial,trial_reward,n_frames,sum_trials_rewards,total_frames):
        now = datetime.now()        
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        log_data_dict =[('Rom',self.rom_name),
        ('Time',dt_string),
        ('Trial', current_trial),
        ('TrialReward', trial_reward),
        ('TrialFrames', n_frames),
        ('SumTrialsRewads', sum_trials_rewards),
        ('TotalFrames', total_frames)]
        super().log(log_data_dict) 
    
   
    


       




