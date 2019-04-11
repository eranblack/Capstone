import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from rfpimp import importances, plot_importances
import seaborn as sns
from sklearn.model_selection import cross_validate
%matplotlib inline
pd.set_option("display.max_columns", 2000)
pd.set_option('precision', 2) #setting the number of decimel points

# reading in the data:
df_full=pd.read_csv('data/Seasons_Stats.csv')
df_full.drop(columns='Unnamed: 0', inplace=True)

#cleaning the data and keeping only data from 2005 and up
df=df_full[df_full['Year']>=2005]
df.drop(labels='blank2', axis=1, inplace=True)
df.drop(labels='blanl', axis=1, inplace=True)
df['Year']=df['Year'].astype(int)
df['Year']=pd.to_datetime(df['Year'], format='%Y')
df['Year']=df['Year'].dt.year



#function for cleaning the data:
def remove_tot(combined):
    tot_player=combined[combined['Tm']=='TOT']['Player'].values
    for name in tot_player:
        temp=combined[combined['Player']==name]
        tot_index=temp[temp['Tm']=='TOT'].index.values
        for idx in tot_index:
            year=temp.loc[idx]['Year']
            year_index=temp[temp['Year']==year].index.values
            year_temp=temp[temp['Year']==year]
            max_year=((year_temp[year_temp['Tm']!='TOT'])['G'].max())
            team=(year_temp[year_temp['G']==max_year])['Tm'].values
            team=str(team)
            team=team[1:-1]
            combined.at[idx,'Tm']=team
            for element in year_index:
                if element==idx:
                    continue
                else:
                    combined.drop(axis=0, index=element, inplace=True)
    return(combined)

df=remove_tot(df)
df['team']=df['Tm']
df.drop(columns='Tm',inplace=True)
df['team']=df['team'].apply(lambda x: x[1:4] if len(x)>3 else x)
df=df.drop_duplicates(subset=['Year','Player'], keep=False)

#creating new features:
df['ppg']=df['PTS']/df['G']
df['ppm']=df['PTS']/df['MP']

#function for adding rolling point average and variance:
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

df=mean_var(df)

    #rearanging the order of columns:
df=df[['Year','team','Player','ppg','ppm','mean_ppg','var_ppg','PTS','Pos','Age','G','GS','MP','PER','TS%','3PAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','OWS','DWS','WS','WS/48','OBPM','DBPM',
            'BPM','VORP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']]
# can load data from data/player_2005.csv and continue from here.
#creating past season data:
df['prev_year']=df['Year']-1
df=pd.merge(df,df, how='left', left_on=['prev_year','Player'], right_on=['Year','Player'])

#removing edge data (players that dont have history)
df=df.dropna(thresh=56)

#filling in missong values of zeroes:
df['3P%_y']=df['3P%_y'].fillna(value=0)
df['FT%_y']=df['FT%_y'].fillna(value=0)
df['3P%_x']=df['3P%_x'].fillna(value=0)
df['FT%_x']=df['FT%_x'].fillna(value=0)
df['ppm_y']=df['ppm_y'].fillna(value=0)
df['MP_y']=df['MP_y'].fillna(value=0)
df['eFG%_y']=df['eFG%_y'].fillna(value=0)
df['FG%_y']=df['FG%_y'].fillna(value=0)
df['ppm_x']=df['ppm_x'].fillna(value=0)

#dropping players who dont have variance of past season :
df=df[~df['var_ppg_y'].isna()]

#Simple model:
X=df[['ppg_y','MP_y','Age_x','FG%_y','FGA_y','eFG%_y','FT%_y','FTA_y','3P%_y','3PA_y','PF_y','mean_ppg_y','var_ppg_y']]
y=df[['ppg_x']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rf_reg = RandomForestRegressor(max_depth=5, random_state=0,n_estimators=200)
rf_reg.fit(X_train,y_train)

#cross validation:
scores=cross_validate(rf_reg,X=X_train,y=y_train,scoring='neg_mean_squared_error')

#error

-scores['test_score'].mean()


#feature Importances:
imp =importances(rf_reg, X_test, y_test) # permutation
viz = plot_importances(imp,width=6, vscale=2)
viz.view()

#plotting:

plt.scatter(y_test.values,y_test.values-y_pred.reshape(-1,1),alpha=0.3, c='orange')
plt.title('y_test vs residuals')
plt.xlabel('y_test')
plt.ylabel('residuals')
