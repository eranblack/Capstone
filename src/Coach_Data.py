# scraping coach data from basketball refrence:
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

%matplotlib inline
pd.set_option("display.max_columns", 2000)
pd.set_option('precision', 2) #setting the number of decimel points

req = requests.get('https://www.basketball-reference.com/coaches/NBA_stats.html')
content=req.content
soup=BeautifulSoup(content)
table=soup.find('table')

# Getting the coach names:
coach_name=[]
for element in table.find_all('a'):
    coach_name.append(element.text)


#scarping all of the coach tables
tables=table.find_all('a')
coach_list=[]
for element in tables:
    loc='https://www.basketball-reference.com/' + element.attrs['href']
    raw_coach_tables=requests.get(loc)
    raw_coach_table_content=raw_coach_tables.content
    soup2=BeautifulSoup(raw_coach_table_content)
    coach_table_content=soup2.find('table')
    coach_list.append(pd.read_html(str(coach_table_content)))


# cleaning and merging coach table:

for idx,element in enumerate(coach_list):
    element[0]['coach_name']=coach_name[idx]
    if idx==0:
        all_coach=element[0]
    else:
        all_coach=all_coach.append(element[0])

# Organizing the data into a dataframe:

df_coach=pd.DataFrame()
df_coach['Team']=all_coach['Playoffs']['Tm']
df_coach['coach_age']=all_coach['Regular Season']['Age']
df_coach['coach_name']=all_coach['coach_name']
df_coach['season']=all_coach['Unnamed: 0_level_0']['Season']
df_coach['playoff_games']=all_coach['Unnamed: 11_level_0']['G']
df_coach['playoff_wins']=all_coach['Unnamed: 12_level_0']['W']
df_coach['playoff_loses']=all_coach['Unnamed: 13_level_0']['L']
df_coach['playoff_W/L%']=all_coach['Unnamed: 14_level_0']['W/L%']
df_coach['season_games']=all_coach['Unnamed: 4_level_0']['G']
df_coach['season_wins']=all_coach['Unnamed: 5_level_0']['W']
df_coach['season_loses']=all_coach['Unnamed: 6_level_0']['L']
df_coach['season_W/L%']=all_coach['Unnamed: 7_level_0']['W/L%']
df_coach['season_position']=all_coach['Unnamed: 9_level_0']['Finish']

df_coach.dropna(axis=0,inplace=True, thresh=6) #dropped assistant coaches which have less than 6 fields filled

# Test weather all the coach data was scraped and no coach is missing --> chack missing
missing=[]
for element in coach_name:
    if element not in df_coach['coach_name'].values:
        missing.append(element)

# removing carreer aggregated rows:
df_coach=df_coach[df_coach['season']!='Career']

# changing the season year from 2017-18 --> 2018
df_coach['season']=df_coach['season'].apply(lambda x: x[:2]+x[-2:]) #code for fixing ther dates


df_coach.drop(labels='Unnamed: 0',inplace=True, axis=1) #removing an old index column thats not needed
# writing the cleaned data to csv:
df_coach.to_csv('data/coach_data_cleaned.csv') #this is the cleaned coach data csv
