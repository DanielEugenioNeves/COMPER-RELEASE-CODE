"""           
     Transition: [st_1,a,r,st,q,done] -> size 14116.  
     Transition: [st_1,a,r,st,q,done] -> size 56452.                        
"""

class FrameTransitionTypes:
        NOT_DEFINED   = 0
        SINGLE_FRAMES = 1
        STAKED_FRAMES = 2

class FrameTransition:
        TYPE = FrameTransitionTypes.NOT_DEFINED        
        ST_W = 0
        ST_H = 0
        ST_L = 0    
        T_LENGTH = 0
        T_N_IDX  = 0
        T_IDX_ST_1 = [0,0]
        T_IDX_A    = 0
        T_IDX_R    = 0
        T_IDX_ST   = [0,0]
        T_IDX_Q    = 0
        T_IDX_DONE = 0

        @staticmethod
        def turn_staked_frames():               
                FrameTransition.TYPE = FrameTransitionTypes.STAKED_FRAMES
                FrameTransition.ST_W = 84
                FrameTransition.ST_H = 84
                FrameTransition.ST_L = 4    
                FrameTransition.T_LENGTH = 56452
                FrameTransition.T_N_IDX  = 56451
                FrameTransition.T_IDX_ST_1 = [0,28224]
                FrameTransition.T_IDX_A    = 28224
                FrameTransition.T_IDX_R    = 28225
                FrameTransition.T_IDX_ST   = [28226,56450]
                FrameTransition.T_IDX_Q    = 56450
                FrameTransition.T_IDX_DONE = 56451

        @staticmethod
        def turn_single_frames():                
                FrameTransition.TYPE = FrameTransitionTypes.SINGLE_FRAMES 
                FrameTransition.ST_W = 84
                FrameTransition.ST_H = 84
                FrameTransition.ST_L = 1    
                FrameTransition.T_LENGTH = 14116
                FrameTransition.T_N_IDX  = 14115
                FrameTransition.T_IDX_ST_1 = [0,7056]
                FrameTransition.T_IDX_A    = 7056
                FrameTransition.T_IDX_R    = 7057
                FrameTransition.T_IDX_ST   = [7058,14114]
                FrameTransition.T_IDX_Q    = 14114
                FrameTransition.T_IDX_DONE = 14115