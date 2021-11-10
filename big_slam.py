#!/usr/bin/env python
# coding: utf-8

# In[10]:
import os
from numpy.fft import fft, ifft, fftshift
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

    
###############################################################################################################
    
#Load experiment data
from scipy.io import loadmat
def load_test_data(region,cap):
    
    #Load .mat dictionary
    file_str = "Experiment_Data/Region_%d_Cap_%d_9600Hz.MAT" % (region, cap)
    mat_dict = loadmat(file_str)
    channels = ['Channel_%d_Data' % i for i in range(2,12)]
    channel_data_list = []
    for i in channels:
        channel_data_list.append(mat_dict[i])
        
    if file_str == 'Experiment_Data/Region_2_Cap_2_9600Hz.MAT': #There is an extra impulse in this test
        dataset = np.array(channel_data_list)[:,0:265000,0]
    else:
        dataset = np.array(channel_data_list)[:,:,0]
    header = ['Channel_%d'% i for i in range(1,11)]
    df = pd.DataFrame(dataset.T,columns = header)
    
    return df


###############################################################################################################
    
#Recurrence plots function

from scipy.spatial.distance import pdist, squareform

def recurrence_plot(data_window,method):
    
    #data is a pandas Dataframe
    #method: [‘euclidean’,’cosine’,’mahalanobis’]
    
    if method != 'mahalanobis':
        rp = squareform(pdist(data_window, method))
    else :
        
        data_cov = data_window.cov().values
        
        rp = squareform(pdist(data_window, method ,data_cov))
    return rp
###############################################################################################################
    
#lead_lag_detector class

from scipy.ndimage import gaussian_filter1d

class lead_lag_detector:
    
    # The init method or constructor
    def __init__(self, data_window ,num_sensors,gaus_param):
        # data_window - pandas DataFrame 
        # gaus_param - value used in gaussian_filter1d for signal smoothing
        if gaus_param == 0:
            self.data_window = data_window.abs().values
        else:
            self.data_window = data_window.abs().apply(lambda x:gaussian_filter1d(x,gaus_param)).values
            
        self.num_sensors = num_sensors 
        self.get_lead_lag_matrix()

    #Cross correlation using fft to convolve signals x and y
    def cross_correlation_using_fft(self,x, y):
        
        f1 = fft(x)
        f2 = fft(np.flipud(y))
        cc = np.real(ifft(f1 * f2))
        return fftshift(cc)

    #Compute the tau time lag shifted values
    def compute_shift(self,x, y):
        
        assert len(x) == len(y)
        c = self.cross_correlation_using_fft(x, y)
        assert len(c) == len(x)
        zero_index = int(len(x) / 2) - 1
        shift = zero_index - np.argmax(c)
        return -shift

    #Compute the lead-lag matrix of input data_window
    def get_lead_lag_matrix(self):
        
        self.lead_lag_matrix = np.zeros((self.num_sensors,self.num_sensors))
        self.lead_lag_matrix_un = np.zeros((self.num_sensors,self.num_sensors))
        self.pulse_lags = np.zeros(self.num_sensors)
        self.pulse_location = 0

        for i in range(self.num_sensors):
            for j in range(self.num_sensors):
                x = self.data_window[:,i]
                y = self.data_window[:,j]
                
                self.lead_lag_matrix_un[i,j] = self.compute_shift(x,y)
                self.lead_lag_matrix[i,j] = np.heaviside(self.compute_shift(x,y),0.5)
              
        self.pulse_lags = self.lead_lag_matrix.mean(axis=1) 
        self.pulse_location = self.lead_lag_matrix.mean(axis=1).argmin()
    
    #Plotting function of lead-lag matrix and its row wise mean
    def plot(self):
        
        
        matplotlib.rcParams.update({'font.size': 12})

        fig3 = plt.figure(constrained_layout = True,figsize=(10,6))
        gs = fig3.add_gridspec(3,3)
        
        #Pulse lag plot
        fig3_ax1 = fig3.add_subplot(gs[:,0])

        fig3_ax1.plot(self.pulse_lags,range(self.num_sensors),'k.-')
        
        fig3_ax1.set_yticks(range(self.num_sensors))
        fig3_ax1.set_ylim(-0.5,self.num_sensors-0.5)
        fig3_ax1.set_yticklabels(['Z%d'%i for i in range(1,self.num_sensors+1)])
        
        fig3_ax1.hlines(self.pulse_location,0,max(self.pulse_lags),'r',linestyles='dashed')
        fig3_ax1.set_title(r'$\mu_A$')
        fig3_ax1.text(max(self.pulse_lags)-0.2 ,self.pulse_location + 0.1,'Minimum')
        
        #Lead_lag matrix plot
        fig3_ax2 = fig3.add_subplot(gs[:,1:3])

        im = fig3_ax2.imshow(self.lead_lag_matrix, interpolation='nearest',cmap='gray',origin='lower')
        fig3_ax2.yaxis.tick_right()
        fig3_ax2.xaxis.tick_bottom()
        fig3_ax2.set_yticks(range(self.num_sensors))
        fig3_ax2.set_yticklabels(['Z%d'%i for i in range(1,self.num_sensors+1)])
        fig3_ax2.set_xticks(range(self.num_sensors))
        fig3_ax2.set_xticklabels(['Z%d'%i for i in range(1,self.num_sensors+1)])
        fig3_ax2.set_title('Lead-lag matrix A')
        fig3.colorbar(im)
###############################################################################################################
    
#Max peak detector class

class max_peak_detector:
    
    # The init method or constructor
    def __init__(self, data_window, num_sensors):
        
        # data_window - pandas DataFrame 
        self.data_window = data_window.abs().values.T
        self.num_sensors = num_sensors
        self.peaks = np.zeros(num_sensors)
        self.get_max_peak()



    def get_max_peak(self):
        
        #Find the time of the max abs peaks for each sensor
        for i,row in enumerate(self.data_window):
            self.peaks[i]= row.argmax()
        #Find the sensor that has its max at the earliest time
        self.max_peak_sensor = self.peaks.argmin()
        
        
    #Plotting function of the local max peaks indicating the minimum
    def plot(self):
        
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['font.size'] = 14
        matplotlib.rcParams['font.family'] = "serif"
       
        fig,ax = plt.subplots(figsize = (10,6))

        ax.plot(self.peaks,np.arange(self.num_sensors),'k.-',markersize=8,label='Max. peaks')
        #ax.vlines(min(self.peaks),0,self.num_sensors,color='red',linestyle='dashed')
        
        ax.text(max(self.peaks)- 10 ,self.max_peak_sensor + 0.1,'Minimum')
        ax.hlines(self.max_peak_sensor,min(self.peaks),max(self.peaks),'r',linestyles='dashed')
        #ax.set_ylabel('Sensor:')
        ax.set_xlabel('Time: [s]')
        ax.set_yticks(range(self.num_sensors))
        ax.set_yticklabels(['Z%d'%i for i in range(1,self.num_sensors+1)])
        
        #ax.text(min(self.peaks) - 2  ,0,'Minimum',rotation=90)
        ax.set_xlim(min(self.peaks) - 3,max(self.peaks)+3)
        plt.legend()
        
        
###############################################################################################################
    
#Wave slamming detection class

from skimage.transform import resize

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten
from pyts import utils
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as scikit_PCA
import matplotlib.pyplot as plt
import umap as umap_decomposition
from sklearn.cluster import DBSCAN

class wave_slam_detector:
    
    # The init method or constructor
    def __init__(self, dataframe):
        
        # dataframe - pandas DataFrame 
        self.df = dataframe


    def feature_extraction(self,window_size,stride,compressed_window_size,recurrence_plot_type):
        
        #recurrence_plot_type - ['euclidean','cosine',mahalanobis']
        #Create a windowed index for discrete overlapping time windows
        self.windowed_index = utils.windowed_view(np.arange(0,self.df.shape[0]).reshape(1, -1),window_size,stride)[0]

        #load pretrained CNN model
        model = VGG16(include_top=False, input_shape=(compressed_window_size, compressed_window_size, 3))
        flat1 = Flatten()(model.layers[-1].output)
        model = Model(inputs=model.inputs, outputs=flat1)

        VGG16_feature_vectors = []

        #Loop through time windows
        for window in self.windowed_index:

            #Raw data window
            img_data = self.df.iloc[window,:]
            #Generate RP
            img = recurrence_plot(img_data,recurrence_plot_type)
            #Compress RP
            img = resize(img,(compressed_window_size,compressed_window_size),anti_aliasing=True)

            #Generate image tensor
            gg = np.zeros((img.shape[0],img.shape[1],3))
            gg[0:img.shape[0],0:img.shape[1],0] = img
            gg[0:img.shape[0],0:img.shape[1],1] = img
            gg[0:img.shape[0],0:img.shape[1],2] = img

            x = np.expand_dims(gg, axis=0)
            #extract feature vector
            img_feature_vector = model.predict(x)[0]
            VGG16_feature_vectors.append(img_feature_vector)
            
        self.feature_vectors = VGG16_feature_vectors

        
    def PCA(self,max_components,plot_variance=True):
        
        pca_model = scikit_PCA(n_components = max_components)
        
        self.pca_transformed = pca_model.fit_transform(self.feature_vectors)
        
        if plot_variance:
            fig,ax = plt.subplots(figsize=(10,5))
            
            plt.plot(np.cumsum(pca_model.explained_variance_ratio_),marker='.',c='k')
            
            plt.xticks(range(max_components))
            ax.set_xticklabels(np.arange(max_components)+1)
            plt.title('PCA Components cumulative  variance');
            plt.xlabel('No. of components');
            plt.ylabel('Variance');
        
    def UMAP(self,n_neighbors,min_dist,plot=True,connect_plot=False):
    
        umap_model = umap_decomposition.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,random_state=18,n_components=2)
        self.umap_transformed = umap_model.fit_transform(self.pca_transformed)
        
        if plot:
            fig,ax = plt.subplots(figsize=(10,5))
            ax.scatter(self.umap_transformed[:,0],self.umap_transformed[:,1],c='k')
            ax.set_title('UMAP decomposition');
        if connect_plot:
            fig,ax = plt.subplots(figsize=(10,5))
            ax.plot(self.umap_transformed[:,0],self.umap_transformed[:,1],'k.-',alpha=0.5)
            ax.set_title('UMAP decomposition');
            
    def clustering(self,eps,min_samples,plot=True):
    
        DBSCAN_model = DBSCAN(eps=eps, min_samples=min_samples).fit(self.umap_transformed)

        self.DBSCAN_feature_labels = DBSCAN_model.labels_ + 1

        if plot:
                fig,ax = plt.subplots(figsize=(10,5))
                ax.scatter(self.umap_transformed[:,0],self.umap_transformed[:,1],c= self.DBSCAN_feature_labels)
                ax.set_title('UMAP decomposition');
                
                for col in np.unique(self.DBSCAN_feature_labels):
                    ax.scatter(self.umap_transformed[self.DBSCAN_feature_labels==col,0]
                              ,self.umap_transformed[self.DBSCAN_feature_labels==col,1]
                              ,label='Cluster %d'%col)
                    
                ax.set_xlabel(r'$U_1$')
                ax.set_ylabel(r'$U_2$')
                
                # Shrink current axis by 20%
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                # Put a legend to the right of the current axis
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                plt.tight_layout()
        
                
    def user_cluster_evaluation(self):

        self.chosen_cluster = []

        fig,ax = plt.subplots(2,1,figsize=(15,6))
        #Plot DBSCAN labels
        ax[0].plot(self.DBSCAN_feature_labels,'.',markersize=10)
        ax[0].set_ylabel('Cluster label')
        ax[0].xaxis.set_visible(False)
        ax[0].set_title('DBSCAN cluster labels');
        ax[0].set_ylim(0,max(self.DBSCAN_feature_labels)+1)
        
        #Plot first variable of original signal
        ax[1].plot(self.df.iloc[:,0].values,linewidth=0.8)
        ax[1].set_title('Channel 1');
        ax[1].set_ylabel(r'Acceleration [$m/s^2$]');
        
    
    def clustering_in_time(self,eps=10,min_samples=1):
        
        DBSCAN_clusters = pd.DataFrame({'DBSCAN_labels' :self.DBSCAN_feature_labels,
                                        'window_index':range(len(self.DBSCAN_feature_labels)),
                                        'window_start':np.zeros(len(self.DBSCAN_feature_labels)),
                                        'window_end':np.zeros(len(self.DBSCAN_feature_labels)),
                                        'impact_number':np.zeros(len(self.DBSCAN_feature_labels))})
        
        #Select only points corresponding to chosen cluster
        clusters_in_time = DBSCAN_clusters[DBSCAN_clusters['DBSCAN_labels']==self.chosen_cluster]
        
        #Cluster the selested cluster group in time
        DBSCAN_model = DBSCAN(eps=eps, min_samples=1).fit(clusters_in_time.values)
        self.DBSCAN_time_labels = DBSCAN_model.labels_ + 1
        
        #Find the that an identified cluster in time starts and ends
        clusters_in_time.loc[:,'window_start'] = [i[0] for i in self.windowed_index[clusters_in_time['window_index']]]
        clusters_in_time.loc[:,'window_end'] = [i[-1] for i in self.windowed_index[clusters_in_time['window_index']]]
        clusters_in_time.loc[:,'impact_number'] = self.DBSCAN_time_labels
        
        #Create dataframe with corresponding impacts
        impulse_list = []

        for impact in clusters_in_time['impact_number'].unique():
            impulse_dict = {'impact_number':[],
                        'start': [],
                        'end':[],
                        'approximate_impact_time':[]
                           }
            df_impact_number = clusters_in_time[clusters_in_time['impact_number']==impact]
            impulse_dict['impact_number'] = impact
            impulse_dict['start'] = df_impact_number['window_start'].values[0]
            impulse_dict['end'] = df_impact_number['window_end'].values[-1]
            impulse_list.append(impulse_dict)

        self.results_df = pd.DataFrame(impulse_list)
        
        for i in range(self.results_df.shape[0]):

            time_window = self.df.iloc[self.results_df.loc[i,'start']:self.results_df.loc[i,'end'],:].abs().values.T

            approx_time = self.results_df.loc[i,'start'] + time_window.mean(axis=0).argmax()

            self.results_df.loc[i,'approximate_impact_time'] = approx_time
            
        return self.results_df
   
    
