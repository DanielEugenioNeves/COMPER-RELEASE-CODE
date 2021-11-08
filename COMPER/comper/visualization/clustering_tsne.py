import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import tensorflow as tf
import keras.backend as K
import pandas as pd
import seaborn as sns
import os

class ClusteringTSNE(object):
    def __init__(self):        
        self.n_components=2
        self.perplexity=30.0
        self.early_exaggeration=12.0
        self.learning_rate=200.0
        self.n_iter=1000
        self.n_iter_without_progress=300
        self.min_grad_norm=1e-07
        self.metric='euclidean'
        self.init='random'
        self.verbose=0
        self.random_state=None
        self.method='barnes_hut'
        self.angle=0.5
        self.n_jobs=None
        self.square_distances='legacy'
        self.q_value =[]        
        self.__initialize()
    
    def __initialize(self):
        self.init="pca"
        self.random_state=42
        self.n_jobs=3
        self.__config_tsne()
    
    def __config_tsne(self):
        self.tsne = TSNE(n_components=2,perplexity=self.perplexity, early_exaggeration=self.early_exaggeration,learning_rate=self.learning_rate,
                         n_iter=self.n_iter, n_iter_without_progress=self.n_iter_without_progress, min_grad_norm=self.min_grad_norm, 
                         metric=self.metric, init=self.init, verbose=self.verbose, random_state=self.random_state, method=self.method, 
                         angle=self.angle, n_jobs=self.n_jobs, square_distances=self.square_distances)

    def clustering(self,frames,convnet,layer_name,q_values,save_img_path="",image_name=""):
        self.q_value = np.array(np.round(q_values,decimals=2))    
        activations=[]
        for i in range(len(frames)):
            keras_function = K.function([convnet.input], [convnet.get_layer(name=layer_name).output])
            result = keras_function([frames[i]])
            activations.append(np.array(result[0]).flatten())
        
        embedded = self.tsne.fit_transform(activations)
        self.plot_embedded(embedded,save_img_path,image_name)
    
    def plot_embedded(self,embedded,save_img_path="",image_name=""):
        #sns.scatterplot(x=embedded[:,0], )
        #c = self.q_value if len(self.q_value) >0 else None
        #plt.scatter(embedded[:,0], y=embedded[:,1],c=c,cmap="jet",alpha=0.8)
        #plt.show()
        
        df_subset = pd.DataFrame()
        df_subset['tsne-2d-one'] = embedded[:,0]
        df_subset['tsne-2d-two'] = embedded[:,1]
        df_subset["q_value"] = self.q_value
        #print("teste")
        #print(df_subset.head(5))
        #df_subset.head(5)
        #plt.figure(figsize=(16,10))
        sns.scatterplot(
            data= df_subset,
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue="q_value",
            #palette=sns.color_palette("hls", df_subset["q_value"].nunique()),
           
        )
        
        if(save_img_path!="" and image_name!=""):
            os.makedirs(save_img_path, exist_ok=True)
            save_img_path=save_img_path+"/"+image_name
            plt.savefig(save_img_path)
        else:
            plt.show()


    

