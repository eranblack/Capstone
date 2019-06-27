import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, silhouette_score
from rfpimp import importances, plot_importances
from rfpimp import plot_corr_heatmap
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 2000)
pd.set_option('precision', 2) #setting the number of decimel points

class ModelP: # defines the class model, gets a year as an input (the year is the year of the prediction, for example using 2012 will train the model on all previous years)
    def __init__(self,year):
        self.year=year
        self.X_train=pd.DataFrame()
        self.y_train=pd.DataFrame()
        self.X_test=pd.DataFrame()
        self.y_test=pd.DataFrame()
        self.label_train=np.array([])
        self.label_test=np.array([])
        self.team=pd.DataFrame([])
        self.y_pred=np.array([])
        self.rf_reg=RandomForestRegressor(max_depth=5, random_state=0,n_estimators=200)

    def fit(self,df): # This method is for fitting the model. It creates clusters and sets up the data for the regression model, and fits the model
        # defining the training data:
        df_train=df[df['Year_x']<self.year]
        mini_train=MiniBatchKMeans(n_clusters=8, batch_size=100, random_state=4)
        X_cluster_train=df_train[['log_mean_ppg_y','ppg_y_std']]
        self.label_train=mini_train.fit_predict(X_cluster_train) #getting the clusters from the KMeans model
        df_train['label']=self.label_train # adding the clusters as a new feature to the training set
        df_train=pd.get_dummies(df_train,columns=['label'])
        df_year=df[df['Year_x']==self.year]

        # defining the test data
        mini=MiniBatchKMeans(n_clusters=8, batch_size=100, random_state=4) #setting the cluster for the test set
        X_cluster_test=df_year[['log_mean_ppg_y','ppg_y_std']]
        self.label_test=mini.fit_predict(X_cluster_test)
        df_year['label']=self.label_test
        df_year=pd.get_dummies(df_year,columns=['label'])
        self.team=df_year['team_x'].copy()
        self.X_train=df_train[['Age_x','ppg_y','log_mean_ppg_y','var_ppg_y','label_0','label_1','label_2','label_3','label_4','label_5','label_6']]
        self.y_train=df_train[['ppg_x']]
        self.X_test=df_year[['Age_x','ppg_y','log_mean_ppg_y','var_ppg_y','label_0','label_1','label_2','label_3','label_4','label_5','label_6']]
        self.y_test=df_year[['ppg_x']]

    # this is the predict method for the model.
    def predict(self,*args): # predicts the player/all players ppg depending on the the *args. Returns the prediction for the player/players
        player_d={}
        self.rf_reg.fit(self.X_train,self.y_train)
        self.y_pred=self.rf_reg.predict(self.X_test)
        if not args:
            all_player=self.X_test.copy()
            all_player['ppg_pred']=self.y_pred.copy()
            all_player['label']=self.label_test.copy()
            all_player['true_ppg']=self.y_test.copy()
            all_player['team']=self.team.copy()
            return (all_player[['Age_x','ppg_y','ppg_pred','true_ppg','label']])
        else:
            for arg in args:
                player_pred=self.X_test.copy()
                player_pred['label']=self.label_test.copy()
                player_pred['ppg_pred']=self.y_pred.copy()
                player_d[arg]=player_pred[player_pred.index==arg+str(self.year)][['Age_x','ppg_y','ppg_pred','label']].copy()
                player_d[arg]['y_test']=self.y_test.copy()
                player_d[arg]=player_d[arg][['Age_x','ppg_y','ppg_pred','y_test','label']]
            return player_d

    # Method for evaluating the model. Compares the MSE score of the model vs the benchmark model
    def score(self):
        mse=mean_squared_error(self.y_test,self.y_pred)
        print('The MSE is : {}'.format(mse))

        benchmark_error=mean_squared_error(self.y_test,self.X_test['ppg_y'])
        print('The benchmark MSE is : {}'.format(benchmark_error))
        return (mse)

    # method for plotting the residuals and hist of the model predictions. Method recieves plot type as 'residuals' or 'hist'
    def plots(self,plot_type='residuals'): # plots --> plot_type= 'residuals' or 'hist'
        hand={}
        self.y_pred.reshape(-1,1)
        temp=pd.DataFrame(index=self.y_test.index, columns=['y_test','y_pred','cluster'])
        temp['y_test']=self.y_test
        temp['y_pred']=self.y_pred
        temp['cluster']=self.label_test
        cmap={0:'red',1:'orange', 2:'green', 3:'blue', 4:'magenta',5:'cyan',6:'black',7:'forestgreen'}
        if plot_type=='residuals':
            for element in temp['cluster'].unique():
                class_temp=temp[temp['cluster']==element]
                hand[element]=plt.scatter(class_temp['y_test'].values,class_temp['y_test'].values-class_temp['y_pred'].values,alpha=0.6, color=cmap[element],marker='.' ,label='clusters {}'.format(element))
            plt.title('y_test vs residuals')
            plt.xlabel('y_test')
            plt.ylabel('residuals')
            plt.legend((hand[0],hand[1],hand[2],hand[3],hand[4],hand[5],hand[6],hand[7]), ('cluster 1','cluster 2','cluster 3','cluster 4','cluster 5','cluster 6','cluster 7'))
            plt.show()
        elif plot_type=='hist':
            fig,axis = plt.subplots(2,4, figsize=(15,10))
            axis=axis.ravel()
            for element in temp['cluster'].unique():
                class_temp=temp[temp['cluster']==element]
                axis[element].hist(class_temp['y_test'].values-class_temp['y_pred'].values, bins=10, color='orange')
                axis[element].set_title('cluster {} residuals'.format(element))
                axis[element].set_xlabel('residuals')
                axis[element].set_ylabel('count')
            plt.tight_layout()
            plt.show()

    #Method for looking and evaluating the clusters the model made.
    def clusters(self): # get the clusters with the residuals and player names
        cluster={}
        clus_resids={}
        c_temp=pd.DataFrame(index=self.y_test.index, columns=['y_test','y_pred','cluster'])
        c_temp['y_test']=self.y_test
        c_temp['y_pred']=self.y_pred
        c_temp['resids']=c_temp['y_test']-c_temp['y_pred']
        c_temp['cluster']=self.label_test
        for clus in c_temp['cluster'].unique():
            cluster[clus]=c_temp[c_temp['cluster']==clus]
            clus_resids[clus]=(cluster[clus]['resids'].mean(),cluster[clus]['resids'].std())
        return cluster, clus_resids


    #Method for evaluating the permutation importances of the model features
    def importance(self): # permutation feature importance using rfpimp library
        imp =importances(self.rf_reg, self.X_test, self.y_test)
        viz = plot_importances(imp,width=6, vscale=2)
        viz.view()
        print(imp)

    #Method for getting the aggregated prediction per team.  Allows precicting the teams points per game.
    def teams(self):
        team_pred={}
        all_player=self.X_test.copy()
        all_player['ppg_pred']=self.y_pred.copy()
        all_player['label']=self.label_test.copy()
        all_player['true_ppg']=self.y_test.copy()
        all_player['team']=self.team.copy()

        all_player=all_player[['ppg_y','ppg_pred','true_ppg','team','label']]
        team_p=all_player.groupby(by='team').sum()
        team_pred['team_prediction']=team_p.copy()
        team_pred['team_pred_MSE']=mean_squared_error(team_p['true_ppg'],team_p['ppg_pred'])
        team_pred['team_bencmark_MSE']=mean_squared_error(team_p['true_ppg'],team_p['ppg_y'])
        return(team_pred)
