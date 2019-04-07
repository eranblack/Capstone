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



# Feature engineering :
df=pd.read_csv('../data/Season_cleaned.csv') # importing the cleaned data

# Creating new features:

df['ppg']=df['PTS']/df['G'] #points per game
df['ppm']=df['PTS']/df['MP'] # points per minute


#creating window for calculating the previous seasons mean and variance on ppg from previous seasons
def mean_var(df):
    agrre=pd.DataFrame(columns=['Player','Year','ppg','mean_ppg','var_ppg'])
    player_list=df['Player'].unique()
    for element in player_list:
        temp=df[df['Player']==element][['Player','Year','ppg']]
        if temp.shape[0]==1:
            temp['mean_ppg']=temp['ppg']
            temp['var_ppg']=np.nan
        else:
            temp['mean_ppg']=temp['ppg'].expanding().mean()
            temp['var_ppg']=temp['ppg'].expanding().var()
        agrre=agrre.append(temp)
    agrre['Year']=agrre['Year'].astype(int)
    df=df.merge(agrre[['Player','Year','mean_ppg','var_ppg']],on=['Year','Player'] ,how='left')
    return(df)

df=mean_var(df) #calling the function which creates the new columns of the previous mean and variance

# changing order of columns:
df=df[['Year','team','Player','ppg','ppm','mean_ppg','var_ppg','PTS','Pos','Age','G','GS','MP','PER','TS%','3PAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','OWS','DWS','WS','WS/48','OBPM','DBPM',
 'BPM','VORP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']]

# adding a previous year column so every observation would have past season stats:
df['prev_year']=df['Year']-1

# mergin on previous season:
df=pd.merge(df,df, how='left', left_on=['prev_year','Player'], right_on=['Year','Player']) # current year is identified by x, and prev year identified by y

df=df.dropna(thresh=56) # drops rows that dont have a previous season data


# imputing zeroes to missing values due to player not shooting/not scoring :
df['3P%_y']=df['3P%_y'].fillna(value=0)
df['FT%_y']=df['FT%_y'].fillna(value=0)
df['3P%_x']=df['3P%_x'].fillna(value=0)
df['FT%_x']=df['FT%_x'].fillna(value=0)
df['ppm_y']=df['ppm_y'].fillna(value=0)
df['MP_y']=df['MP_y'].fillna(value=0)
df['eFG%_y']=df['eFG%_y'].fillna(value=0)
df['FG%_y']=df['FG%_y'].fillna(value=0)
df['ppm_x']=df['ppm_x'].fillna(value=0)

#removing players who dont have previous year variance
df=df[~df['var_ppg_y'].isna()]

# adding more features to the dataframe:

df['log_mean_ppg_x']=np.log(df['mean_ppg_x'])
df['log_mean_ppg_y']=np.log(df['mean_ppg_y'])
df['ppg_x_std']=np.sqrt(df['var_ppg_x'])
df['ppg_y_std']=np.sqrt(df['var_ppg_y'])

########## ENTER THE PLOT OF THE DISTRIBUTION OF THE LOG(MEAN PPG) VS STD(PPG)


# changing the index of the data frame to the player name and year:

df_index=df['Player'] + df['Year_x'].astype(str)
dfi=df.set_index(df_index,drop=True,inplace=False)

dfi.to_csv('../data/Season_Featurized.csv')
