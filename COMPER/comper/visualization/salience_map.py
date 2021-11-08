import matplotlib.pyplot as plt
#from tf_keras_vis.visualization import visualize_saliency
#from tf_keras_vis.utils import utils
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore
from keras import activations
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
#tf.enable_eager_execution()


class SalienceMap(object):
    def __init__(self,last_dense_idx=None):        
        self.last_dense_idx = last_dense_idx 
    
    def apply_salience_map(self,frame,convnet,save_img_path="",image_name="",smooth=True,real_frame=[]):        
        gaus= None
        grads = self.__generate_salience(frame,convnet)        
        if(smooth):
            gaus  = self.__smooth_overlay(grads)
        highlight_grads =np.array(grads,copy=True)
        highlight_grads = self.__highlight(highlight_grads)        
        self.__save_img(frame,grads,highlight_grads,gaus,save_img_path,image_name,real_frame)

    def __generate_salience(self,frame,convnet):               
        _frame = tf.Variable(frame, dtype=float)
        with tf.GradientTape() as tape:
            pred = convnet(_frame)
            #class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
            class_idxs_sorted = tf.argsort(pred,axis=-1,direction='ASCENDING',stable=False,name=None)
            q_value = pred[0][class_idxs_sorted[0]]
            
        grads = tape.gradient(q_value, _frame)
        dgrad_abs = tf.math.abs(grads)
        dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
        arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
        grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
        tf.disable_eager_execution()        
        return grad_eval

    def __highlight(self,salience_map):        
        for i in range(salience_map.shape[0]):
            for j in range(salience_map.shape[1]):
                for value in ([0.75,0.90,1.00]):
                    if(salience_map[i][j]<0.74):
                        salience_map[i][j]=0.0
                    elif salience_map[i][j] < value:
                        salience_map[i][j] = value
                        break
        return salience_map
    
    def __smooth_overlay(self,grads):
        gaus = ndimage.gaussian_filter(grads[:,:], sigma=5)
        return gaus

    def __save_img(self,frame,salience_map,highlight_salience_map,gaus=[],save_img_path="",image_name="",real_frame=[]):
        frame = frame.reshape(84,84)
        frame = frame+highlight_salience_map                
        if (gaus.any() and real_frame.any()):
            fig, axes = plt.subplots(1,4,figsize=(84,84))
            axes[0].imshow(real_frame)           
            axes[1].imshow(frame)
            axes[2].imshow(salience_map,cmap="gray",alpha=0.8)
            axes[3].imshow(gaus,cmap="jet",alpha=0.8)

        elif gaus.any():       
            fig, axes = plt.subplots(1,3,figsize=(84,84))
            axes[0].imshow(frame)
            axes[1].imshow(salience_map,cmap="gray",alpha=0.8)
            axes[2].imshow(gaus,cmap="jet",alpha=0.8)                                
            #i = axes[3].imshow(gaus,cmap="jet",alpha=0.8)            
            #fig.colorbar(i)                        
        else:       
            fig, axes = plt.subplots(1,2,figsize=(84,84))
            axes[0].imshow(frame)
            axes[1].imshow(salience_map,cmap="jet",alpha=0.8)                       
                   
        #plt.show()
        if(save_img_path!="" and image_name!=""):
            os.makedirs(save_img_path, exist_ok=True)
            save_img_path=save_img_path+"/"+image_name
            fig.savefig(save_img_path)
           
        plt.close()
        plt.close("all")
        plt.clf()
    
    def __save_visualization_data(self,real_frame,frame,action,q_value,reward,activation):
        temp=0
    
    
        
        
        


    