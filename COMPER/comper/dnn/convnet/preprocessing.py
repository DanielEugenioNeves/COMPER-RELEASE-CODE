import numpy as np
import cv2
#import tensorflow as tf
from comper.config.transitions import FrameTransition as ft
from comper.config.transitions import FrameTransitionTypes as ft_types
class Preprocessor(object): 
    
    """def gray_preprocess_obs_ram(self,state):
        self.state_processor = tf.image.resize(state, (ft.ST_W, ft.ST_H), method=tf.image.ResizeMethod.MITCHELLCUBIC)
        return self.state_processor"""

    def gray_preprocess_obs_ram(self,obs):
        try:
            obs = cv2.resize(obs, dsize=(ft.ST_W, ft.ST_H), interpolation=cv2.INTER_AREA)
            obs = np.asarray(obs)
            if ft.TYPE == ft_types.SINGLE_FRAMES:                
                obs = obs.reshape([-1,ft.ST_W, ft.ST_H,1])
            elif ft.TYPE == ft_types.STAKED_FRAMES:
                obs = obs.reshape([ft.ST_W, ft.ST_H,1])

            #obs = obs.astype(np.float32) / 255.                
            return obs
        except Exception as isnt:
           print(type(isnt))
           print(isnt)
           raise(isnt)
    
