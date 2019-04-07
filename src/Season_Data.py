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

# cleaning up the seasonal data I got from Kaggle :

#loading the original data:
df_full=pd.read_csv('../data/Seasons_Stats.csv')
# cleaning the data:
df_full.drop(columns='Unnamed: 0', inplace=True)
# removing the data prior to year 2005:
df=df_full[df_full['Year']>=2005]
# removing balnk colimns, changing the year to a date time :
df.drop(labels='blank2', axis=1, inplace=True)
df.drop(labels='blanl', axis=1, inplace=True)
df['Year']=df['Year'].astype(int)
df['Year']=pd.to_datetime(df['Year'], format='%Y')
df['Year']=df['Year'].dt.year

# function for removing team=TOT for players who switched team in the middle fo the season.
def remove_tot(df_tot):
    tot_player=df_tot[df_tot['Tm']=='TOT']['Player'].values
    for name in tot_player:
        temp=df_tot[df_tot['Player']==name]
        tot_index=temp[temp['Tm']=='TOT'].index.values
        for idx in tot_index:
            year=temp.loc[idx]['Year']
            year_index=temp[temp['Year']==year].index.values
            year_temp=temp[temp['Year']==year]
            max_year=((year_temp[year_temp['Tm']!='TOT'])['G'].max())
            team=(year_temp[year_temp['G']==max_year])['Tm'].values
            team=str(team)
            team=team[1:-1]
            df_tot.at[idx,'Tm']=team
            for element in year_index:
                if element==idx:
                    continue
                else:
                    df_tot.drop(axis=0, index=element, inplace=True)
    return(df_tot)
df=remove_tot(df)
#renaming columns:
df['team']=df['Tm']
df.drop(columns='Tm',inplace=True)

# fixing the team names that cosist more than 3 characters:
df['team']=df['team'].apply(lambda x: x[1:4] if len(x)>3 else x)

# Removing duplicates:
df=df.drop_duplicates(subset=['Year','Player'], keep=False)

# write cleaned data to csv:

df.to_csv('../data/Season_cleaned.csv')
