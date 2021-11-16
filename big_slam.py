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
    
    '''
    Loads test data from the associated file it is stored in.
    
    Parameters
    ----------
    region: int
        The chosen region according to the experimental test.
        
    cap: int
        The chosen cap number according to the experimental test.
    
    Returns
    -------
    df: pandas.DataFrame
        A dataframe containing the experimental data with the columns corresponding to the measured channels.
    
    '''
    
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
    
    '''
    Given a matrix containing data of multivariate signal, create a recurrence plot according 
    to a specific distance metric.
    
    Parameters
    ----------
    data_window: pandas.DataFrame
        A dataframe representing a m by t matrix of m variables of a multivariate signal.
        
    method: str
        The distance metric to use. The distance function can be 'euclidean','cosine','mahalanobis' 
        or any of the methods implemented in the scipy.spatial.distance package.
    
    Returns
    -------
    rp: numpy.array
        A m by m matrix representing the recurrence plot of the given data window.
    '''
    
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
    
    '''
    A class representing the lead lag detector that uses the lead lag relationships between 
    variables in order to determine which variable leads the rest.
    
    ...
    
    Attributes
    ----------
    
    data_window: pandas.DataFrame   
        A dataframe representing a m by t matrix of m variables of a multivariate signal.
        
    num_sensors: int
        The number of sensors/variables in the given data window.
        
    lead_lag_matrix: numpy.array   
        A m by m pairwise matrix representing if a variable lags [0], equal [0.5] or leads [1] 
        another variable.
        
    lead_lag_matrix_un: numpy.array    
        A m by m pairwise matrix representing the time sample difference between variables.
        
    pulse_lags: numpy.array   
        A m array that is the rowise mean of the lead_lag_matrix.
       
    pulse_location: int 
        The m value corresponding to the variable that leads all the others in the given data window.
        
        
    Methods
    -------
    cross_correlation_using_fft(x, y):    
        Computes the cross correlation function between the variables x and y using multiplication 
        in the Fourier domain.
        
     compute_shift(x,y):    
         Computes the number of time samples that a variable needs to be shifted to the maximum of the 
         cross correlation function.
         
     get_lead_lag_matrix():    
         Computes the lead lag matrix, thresholding it and from this calculating the leading variable.
        
      plot(matrix_type):     
          Plots the calculated lead lag matrix with its associated row wise mean and indicates the leading
          variable.

    '''
    
    
    def __init__(self, data_window ,num_sensors,gaus_param):
        
        '''
        Constructs all the necessary attribute for the lead_lag_detector object

        Parameters
        ----------
        data_window: pandas.DataFrame
            A dataframe representing a m by t matrix of m variables of a multivariate signal.

        num_sensors: int
            The number of sensors/variables in the given data window.
            
        gaus_param: int  
            The parameter used in the gaussian_filter1d() smoothing function.

        '''
        
        if gaus_param == 0:
            self.data_window = data_window.abs().values
        else:
            self.data_window = data_window.abs().apply(lambda x:gaussian_filter1d(x,gaus_param)).values
            
        self.num_sensors = num_sensors 
        self.get_lead_lag_matrix()

    def cross_correlation_using_fft(self,x, y):
        
        f1 = fft(x)
        f2 = fft(np.flipud(y))
        cc = np.real(ifft(f1 * f2))
        return fftshift(cc)

    def compute_shift(self,x, y):
        
        assert len(x) == len(y)
        c = self.cross_correlation_using_fft(x, y)
        assert len(c) == len(x)
        zero_index = int(len(x) / 2) - 1
        shift = zero_index - np.argmax(c)
        return -shift


    def get_lead_lag_matrix(self):
        
        # initialise all associated variables
        self.lead_lag_matrix = np.zeros((self.num_sensors,self.num_sensors))
        self.lead_lag_matrix_un = np.zeros((self.num_sensors,self.num_sensors))
        self.pulse_lags = np.zeros(self.num_sensors)
        self.pulse_location = 0

        #Compute the pairwise lead lag relationships for the m by m leag lag matrix
        for i in range(self.num_sensors):
            for j in range(self.num_sensors):
                x = self.data_window[:,i]
                y = self.data_window[:,j]
                
                self.lead_lag_matrix_un[i,j] = self.compute_shift(x,y)
                
                #Theshold the time sample length to cross-correlation maximum using the 
                #heaviside function according to lags [0], equal [0.5] or leads [1] 
                
                self.lead_lag_matrix[i,j] = np.heaviside(self.compute_shift(x,y),0.5)
              
        self.pulse_lags = self.lead_lag_matrix.mean(axis=1) 
        self.pulse_location = self.lead_lag_matrix.mean(axis=1).argmin()
    
    def plot(self, matrix_type = 'thesholded' ):
        
        '''
        Plots the calculated lead lag matrix with its associated row wise mean and indicates the leading
        variable.

        Parameters
        ----------
        matrix_type: str

            The matrix type to use, can be 'thesholded'- self.lead_lag_matrix 
            or 'unthresholded' - self.lead_lag_matrix_un.
            
        Returns
        -------
        
        A plot of the chosen lead lag matrix with the associated row wise mean and leading variable 
        indicator.

        '''
        
        if matrix_type == 'thresholded':
            chosen_matrix = self.lead_lag_matrix
        if matrix_type == 'unthresholded':
            chosen_matrix = self.lead_lag_matrix_un
        
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

        im = fig3_ax2.imshow(chosen_matrix, interpolation='nearest',cmap='gray',origin='lower')
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
    
    '''
    A class representing the max peak detector that uses the time that a variable as its maximum
    compared to the maximums of all other variables in order to detect the leading variable.
    
    ...
    
    Attributes
    ----------
    
    data_window: numpy.array
        A m by t matrix representing a time window of a multivariate signal.
        
    num_sensors: int
        The number of sensors/variables in the given data window.
        

    peaks: numpy.array
        A m length array containing the time sample that an individual variable has had its maximum
        in the given data window.
        
    pulse_location: int
        The m value corresponding to the variable that leads all the others in the given data window.
        
    
    Methods
    -------
         
     get_max_peak():
         Computes the time that each m variable has its maximum value, then uses the relative timings to
         detect the leading variable.
        
      plot():
          Plots the calculated peaks and the time they occur in the given data window, indicating the leading
          variable.

    '''
    
    def __init__(self, data_window, num_sensors):
        
        '''
        Constructs all the necessary attribute for the lead_lag_detector object.

        Parameters
        ----------
        
        data_window: pandas.DataFrame
            A dataframe representing a m by t matrix of m variables of a multivariate signal.

        num_sensors: int
            The number of sensors/variables in the given data window.

        '''
        self.data_window = data_window.abs().values.T
        self.num_sensors = num_sensors
        self.peaks = np.zeros(num_sensors)
        self.get_max_peak()



    def get_max_peak(self):
        
        #Find the time of the max absolute peaks for each sensor
        for i,row in enumerate(self.data_window):
            self.peaks[i]= row.argmax()
            
        #Find the sensor that has its max at the earliest time
        self.pulse_location = self.peaks.argmin()
        
        
    def plot(self):
        
        '''
        Plots the calculated peaks and the time they occur in the given data window, indicating the leading
        variable.
        '''
        
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['font.size'] = 14
        matplotlib.rcParams['font.family'] = "serif"
       
        fig,ax = plt.subplots(figsize = (10,6))

        ax.plot(self.peaks,np.arange(self.num_sensors),'k.-',markersize=8,label='Max. peaks')
        
        ax.text(max(self.peaks)- 10 ,self.pulse_location + 0.1,'Minimum')
        ax.hlines(self.pulse_location,min(self.peaks),max(self.peaks),'r',linestyles='dashed')
   
        ax.set_xlabel('Time: [s]')
        ax.set_yticks(range(self.num_sensors))
        ax.set_yticklabels(['Z%d'%i for i in range(1,self.num_sensors+1)])
        
        ax.set_xlim(min(self.peaks) - 3,max(self.peaks)+3)
        plt.legend()
        
        
###############################################################################################################
    
#Wave slamming detection class

#Import necessary libraries and functions

from skimage.transform import resize
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
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
    
    
    '''
    A class representing the wave slam detector that splits a given multivariate signal into discrete time
    windows, converts each window into an image either a type of Recurrence plot or simply just the time
    window. Features are then extracted from the converted images. The resulting high dimensional feature vectors
    are reduced in dimension using PCA and UMAP. The time windows are clustered in the reduced dimensional space.
    The found clusters are mapped back to the time domain and the associated label of interested is chosen. This 
    chosen label is clustered in time quantifying the instances of the desired events. 
    ...
    
    Attributes
    ----------
    
    df: pandas.DataFrame
        A dataframe representing a m by t matrix of m variables of a multivariate signal.
        
    windowed_index: numpy.array
        A w length array containing the indices for each time window.
    
    feature_vectors: numpy.array
        A w by f matrix where f is the number of extracted features and w is the number of time windows.
        
    pca_model: sklearn.decomposition._pca.PCA
        The PCA model created for the associated max_components number of chosen principle components.
        
    pca_transformed: numpy.array
        A w by p matrix where p is the number of principle components used in the pca_model and w is the
        number of time windows.
    
    umap_model: umap.umap_.UMAP
        The UMAP model created for the associated n_neighbors and min_dist parameters used.
        
    umap_transformed: numpy.array
        A w by 2 matrix where w is the number of time windows, representing the time windows in their features
        in the UMAP reduced two dimensional space.
        
    DBSCAN_feature_labels: numpy.array
        A w length array where w is the number if time windows, of the found DBSCAN cluster labels in the UMAP
        reduced space.
        
     chosen_cluster: int
         After user evaluation this the cluster associated with the phenomenon of interest.
         
     DBSCAN_time_labels: numpy.array
         A w length array where w is the number if time windows, of the found DBSCAN cluster labels of the chosen
         cluster in the time domain.
     
     results_df: pandas.DataFrame
         A dataframe containing the information relating to the found impacts. It contains the number of the impact found,
         the time sample index of the start and ending time window the impact occurred in and the approximate time sample 
         within this time range that the impact occured.
    
    Methods
    -------
    
    feature_extraction(window_size,stride,compressed_window_size,image_type): 
    
        Splits the given multivariate signal into time windows according to the given window_size and stride parameters.
        The time windows are converted to an image according to the image_type string. The images are compressed and 
        passed through the pretrained VGG16 CNN. The resulting feature vectors per image are stored in a matrix.
        
    PCA(max_components):
    
        Decomposes the feature vectors found through the CNN to max_components number of principle components.
        
    UMAP(n_neighbors,min_dist):
    
        Decomposes the PCA reduced feature vectors found into a two dimensionsal space using the UMAP algorithm.
        
     clustering(eps,min_samples):
     
         Clusters the time windows represented in the UMAP feature space using the DBSCAN algorithm.
         
     user_cluster_evaluation():
     
         Plots the found cluster labels corresponding to the time domain of the original signal, allowing the user
         to compare the cluster labels to the original signal.
         
     clustering_in_time():
     
         Clusters the chosen cluster label in time using the DBSCAN algorithm. Extracts the time at which each cluster 
         in time starts and ends. The approximate impact time within this time range is also calculated. These values
         are then stored in a pandas.DataFrame.

    '''
    
    def __init__(self, dataframe):
        
        '''
        Constructs all the necessary attribute for the wave_slam_detector object.

        Parameters
        ----------
        
        data_window: pandas.DataFrame
            A dataframe representing a m by t matrix of m variables of a multivariate signal.

        '''
        self.df = dataframe


    def feature_extraction(self,window_size,stride,compressed_window_size,image_type):
        
        '''
        Given a multivariate signal, segment it into discrete time windows of a specfied size and
        at a set stride. Convert these time windows to images using the recurrence_plot() or use
        the given multivariate data window as an image. Pass these images through a the pre-
        trained VGG16 CNN, extracting the final max-pooling layer.

        Parameters
        ----------
        window_size: int
            The number of sample in each time window.

        stride: int
            The number of samples between each time window.
            
        compressed_window_size: int
            The size of the image (number of pixels n by n) passed to the CNN.
            
        image_type:
            The type of image the time window should be converted to. If the recurrence plot
            representation is used the 'euclidean','cosine' or 'mahalanobis' method is passed. If
            the time window is to be used as an image then 'none' is passed.

        '''
        
        #Create array of indices for each time window 
        self.windowed_index = utils.windowed_view(np.arange(0,self.df.shape[0]).reshape(1, -1),window_size,stride)[0]

        #set the input image size for the CNN
        if compressed_window_size == 0:
            cnn_window_size = window_size
        else:
            cnn_window_size = compressed_window_size
                
        #load pretrained CNN model
        model = VGG16(include_top=False, input_shape=(cnn_window_size, cnn_window_size, 3))
        flat1 = Flatten()(model.layers[-1].output)
        model = Model(inputs=model.inputs, outputs=flat1)

        #initialise the feature vector
        VGG16_feature_vectors = []

        #Loop through time windows
        for window in self.windowed_index:

            #Raw data window
            img_data = self.df.iloc[window,:]
            
            #Generate image representing the time window
            if image_type == 'none':
                img = resize(img_data.abs().values,(cnn_window_size,cnn_window_size),anti_aliasing=True)
            else:
                img = recurrence_plot(img_data,image_type)
            
            #Compress time window
            if compressed_window_size != 0:
                img = resize(img,(compressed_window_size,compressed_window_size),anti_aliasing=True)

            #Generate image tensor
            gg = np.zeros((img.shape[0],img.shape[1],3))
            gg[0:img.shape[0],0:img.shape[1],0] = img
            gg[0:img.shape[0],0:img.shape[1],1] = img
            gg[0:img.shape[0],0:img.shape[1],2] = img

            x = np.expand_dims(gg, axis=0)
            #Apply VGG16 image preprocessing
            x = preprocess_input(x)
            #extract feature vector
            img_feature_vector = model.predict(x)[0]
            VGG16_feature_vectors.append(img_feature_vector)
            
        #Create feature vector attribute
        self.feature_vectors = VGG16_feature_vectors

        
    def PCA(self,max_components,plot_variance=False):
        
        '''
        Decomposes the feature vectors found through the CNN to max_components number of principle components.

        Parameters
        ----------
        max_components: int
            The number of principle components used in the PCA model.

        plot_variance: bool
            States if the cumulative variance plot for the model should be visualised.
       
         '''
        
        # Initialise PCA model
        self.pca_model = scikit_PCA(n_components = max_components)
        
        # Fit and transform the feature vectors to the PCA domain
        self.pca_transformed = self.pca_model.fit_transform(self.feature_vectors)
        
        #Plot the cumulative variance
        if plot_variance:
            
            fig,ax = plt.subplots(figsize=(10,5))
            plt.plot(np.cumsum(self.pca_model.explained_variance_ratio_),marker='.',c='k')
            plt.xticks(range(max_components))
            ax.set_xticklabels(np.arange(max_components)+1)
            plt.title('PCA Components cumulative  variance');
            plt.xlabel('No. of components');
            plt.ylabel('Variance');
        
    def UMAP(self,n_neighbors,min_dist,plot=False,connect_plot=False):
        
        '''
        Projects the PCA reduced domain feature vectors to two dimensions using the UMAP algorithm.

        Parameters
        ----------
        n_neighbors: int
            The number of neighbours connected in the UMAP algorithm.

        min_dist: int
            The minimum distance between points in the reduced feature space.
            
        plot: bool
            States if the UMAP reduced feature space should be visualised.
            
        connect_plot: bool
            States if the UMAP reduced feature space should be visualised where each point is connected
            to the next point in time.
         '''
        # Initialise the UMAP model
        self.umap_model = umap_decomposition.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,random_state=18,n_components=2)
        
        # Fit and transform the PCA reduced feature vectors using the UMAP algorithm
        self.umap_transformed = self.umap_model.fit_transform(self.pca_transformed)
        
        #Plot the UMAP reduced feature space
        if plot:
            fig,ax = plt.subplots(figsize=(10,10))
            ax.scatter(self.umap_transformed[:,0],self.umap_transformed[:,1],c='k',marker='.')
            ax.set_title('UMAP decomposition');
        if connect_plot:
            fig,ax = plt.subplots(figsize=(12,10))
            ax.plot(self.umap_transformed[:,0],self.umap_transformed[:,1],'k.-',alpha=0.5)
            ax.set_title('UMAP decomposition');
            
    def clustering(self,eps,min_samples,plot=False):
        
        '''
        Clusters the time windows represented in the UMAP feature space using the DBSCAN algorithm.
        
        Parameters
        ----------
        
        eps: int
            The radius, size of neighbourhood around each point in the DBSCAN algorithm.
            
        min_samples:
            The minimum number of samples in a neighbourhood to be considered a 'core' point.
        
        plot: bool
            States if the UMAP reduced feature space should be visualised, coloured with the found
            DBSCAN cluster labels.
            
        '''
        
        #Initialise the DBSCAN algorithm
        DBSCAN_model = DBSCAN(eps=eps, min_samples=min_samples).fit(self.umap_transformed)

        # Create the feature space label attribute
        self.DBSCAN_feature_labels = DBSCAN_model.labels_ + 1

        # Plot the UMAP feature space with the found DBSCAN cluster labels
        if plot:
                fig,ax = plt.subplots(figsize=(10,5))
                ax.set_title('UMAP decomposition');
                
                for col in np.unique(self.DBSCAN_feature_labels):
                    ax.scatter(self.umap_transformed[self.DBSCAN_feature_labels==col,0]
                              ,self.umap_transformed[self.DBSCAN_feature_labels==col,1]
                              ,marker='.'
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
        
        '''
        Plots the found cluster labels corresponding to the time domain of the original signal, allowing the user
        to compare the cluster labels to the original signal.
  
        '''
        
        # Create empty chosen cluster attribute
        self.chosen_cluster = []

        
        fig,ax = plt.subplots(2,1,figsize=(15,6))
        
        #Plot DBSCAN labels
        ax[0].plot(self.DBSCAN_feature_labels,'k.',markersize=10)
        ax[0].set_ylabel('Cluster label')
        ax[0].xaxis.set_visible(False)
        ax[0].set_title('DBSCAN cluster labels');
        ax[0].set_ylim(0,max(self.DBSCAN_feature_labels)+1)
        ax[0].set_xticks(np.arange(0,max(self.DBSCAN_feature_labels)))
        
        #Plot first variable of original signal
        ax[1].plot(self.df.iloc[:,0].values,'k',linewidth=0.8)
        ax[1].set_title('Channel 1');
        ax[1].set_ylabel(r'Acceleration [$m/s^2$]');
        
    
    def clustering_in_time(self,eps=2,min_samples=1):
        
        '''
        A dataframe containing the information relating to the found impacts. It contains the number of the impact found,
        the time sample index of the start and ending time window the impact occurred in and the approximate time sample 
        within this time range that the impact occured.

        Parameters
        ----------
        eps: int
            The radius, size of neighbourhood around each point in the DBSCAN algorithm.
            
        min_samples:
            The minimum number of samples in a neighbourhood to be considered a 'core' point.
        
        Returns
        -------
        results_df: pandas.DataFrame
            A dataframe containing the found 'impact_number', its start and end time sample corresponding 
            to the time windows the impact occurred in, and the 'approximate_impact_time_sample' that the 
            impact occured at.
        '''
        
        DBSCAN_clusters = pd.DataFrame({'DBSCAN_labels' :self.DBSCAN_feature_labels,
                                        'window_index':range(len(self.DBSCAN_feature_labels)),
                                        'window_start':np.zeros(len(self.DBSCAN_feature_labels)),
                                        'window_end':np.zeros(len(self.DBSCAN_feature_labels)),
                                        'impact_number':np.zeros(len(self.DBSCAN_feature_labels))})
        
        #Select only points corresponding to chosen cluster
        clusters_in_time = DBSCAN_clusters[DBSCAN_clusters['DBSCAN_labels'] == self.chosen_cluster]
        
        #Cluster the selected cluster group in time
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
                        'start_time_sample': [],
                        'end_time_sample':[],
                        'approximate_impact_time_sample':[]
                           }
            df_impact_number = clusters_in_time[clusters_in_time['impact_number']==impact]
            impulse_dict['impact_number'] = impact
            impulse_dict['start_time_sample'] = df_impact_number['window_start'].values[0]
            impulse_dict['end_time_sample'] = df_impact_number['window_end'].values[-1]
            impulse_list.append(impulse_dict)

        self.results_df = pd.DataFrame(impulse_list)
        
        for i in range(self.results_df.shape[0]):

            time_window = self.df.iloc[self.results_df.loc[i,'start_time_sample']:self.results_df.loc[i,'end_time_sample'],:].abs().values.T

            approx_time = self.results_df.loc[i,'start_time_sample'] 
            + time_window.mean(axis=0).argmax()

            self.results_df.loc[i,'approximate_impact_time_sample'] = approx_time
            
        return self.results_df
   
    
