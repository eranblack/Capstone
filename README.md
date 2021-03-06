# Predicting NBA Player performance
<p align="center">
    <img src="Images/Header.jpg" alt="header" height="350" width="350">
 </p>

## Motivation

- NBA teams spend ~100 million dollars a year on players
- Player performance can impact the teams revenue
- Knowing how a player is going to perform can help the team make wise business decisions

## Data

Data Collection:
- player data collected from kaggle https://www.kaggle.com/drgilermo/nba-players-stats
- coach data was scraped from https://www.basketball-reference.com/coaches/
    - Scraped the data via BeautifulSoup
    - Scraping code is located in the Coach_Data.py

Data cleaning:
- merged the coach and the player datasets
- removed all data prior to 2005
- removed duplicates and NaN's
- code for data cleaning in Coach_Data.py, Season_Data.py, featurized_season_data.py

## Part 1 - Data Exploration and initial modeling

<p> <img src="Images/Investigation.jpg" alt="Investigation" align="right"> </p>

I chose a few features for the initial model, and I was particularly interested in assessing the coach feature effect.

Chosen features:
 - Player Age
 - Player previous season ppg
 - Coach
 - previous season minutes played

 ### model1:
 - Created a random forest regressor model that takes in the features discussed above and predicts player ppg
 - I chose to use a limited amount of player features so it would be easier to identify if the coach effects the prediction
 - Created permutation feature importance to see which features are important
 - Tested the feature importance, see figure:
<p> <img src="plots/model1_FI.png" alt="model1_FI" align="right"> </p>

This result shows that the coach isn't a good feature for the model, this doesn't make sense so I decided to investigate further.
### Is there a coach effect?
<p> <img src="Images/Coach_Pop.jpg" alt="Coach_Pop" align="center" height="400" width="550"> </p>


In order to understand this result I tried running the model again after normalizing the data based on player -->
normalized_ppg = player_ppg/max_player_ppg
I did this so players that have a high ppg average don't "hide" the coach effect.

#### model 2 - understanding coach effect:
- created a random forest model with normalized values
- Used the same features I used in model 1
<!-- - for this model the MSE is 0.044 whereas the benchmark is 0.081 -->
- feature importance is still showing the coach has very limited effect
<p align="center">
    <img src="plots/model2_FI.png" alt="alternate text">
 </p>
 Based on this result I decided to further investigate the coach effect:

#### model 3 - understanding coach effect:
 In this model I tried to measure coach effect but on a team level and not a player level:
 - Tried to predict a team W/L % ratio based on previous season W/L% and coach.
 - Created a random forest model which got a MSE 0.0165 whereas the benchmark got 0.0168
 <p align="center">
     <img src="plots/model3_FI.png" alt="alternate text">
 </p>
 <!-- <p align="center">
      <img src="plots/model3_resids.png" alt="alternate text">
 </p>

 <p align="center">
       <img src="plots/model3_bar.png" alt="alternate text">
 </p> -->

 - again the result show that the significance of the coach is low

 Based on these results I decided to set up an hypothesis test that will determine the significance level of the coach.

## part 2 - Coach Hypothesis testing
Set up an Hypothesis test for the coach effect:

- Null Hypothesis --> H<sub>0</sub> - coach has no effect --> W/L% stays the same after team changes a coach
- Alternative Hypothesis --> H<sub>a</sub> - team W/L% changes when a new coach comes
- Significance level &#x3B1; = 5%
- Ran the test comparing between W/L ratio between a season after a coach change and before

Assumptions:
- One game is a Bernoulli trial --> G ~ Bernoulli(P), E[G]=P
- One season of N games is a Binomial distribution --> S ~ Bin(N,P), E[S]=N*P
- S&#772; = S/N , E[S&#772;] = P
- D = S&#772;<sub>after</sub> - S&#772;<sub>before</sub> --> CLT --> D ~ (0,&#x3C3;)
<p align="center">
    <img src="plots/rv_dist.png" alt="rv_dist" style="width:40%">  <img src="plots/Hypo_dist.png" alt="Hypo_dist" style="width:40%">
</p>

Result of a 2 sided t-test:
- p-value = 0.00051 --> reject the null hypothesis --> cant rule out the effect of the coach -->
  needs further investigation
- According to these results I decided to continue optimizing the model without using the coach a feature

## part 3 - Model optimization:
<p> <img src="Images/Optimization.jpg" alt="model1_FI" height="75" width="75"align="right"> </p>

- Tested different features to include in the model, most of the features didn't contribute
    - Benchmark MSE = 9.82
    - Model MSE = 9.328
- Ran a grid search on my random forest for hyper parameter optimization
    - max depth 3 --> 5
    - n_estimators 100 -->200
    - model MSE = 8.896
- Added two features rolling mean ppg and rolling variance ppg that include previous seasons
    - Model MSE = 8.77
- Added KMeans clustering to differentiate between players with high mean / low mean / high variance / low variance
    - Model MSE = 8.744
    - cluster figure:
    <p align="cener">
        <img src="plots/KMeans_minibatch.png" alt="alternate text">
     </p>

- Tried using a GradientBoost model and a AdaBoost model --> bigger error
- Removed features with low permutation importance:
    - Model MSE = 8.676 --> final model
    - Key features:
        - Previous year ppg
        - Cluster (8 clusters, label0 - label6)
        - Age
        - log mean ppg
        - var ppg

## Part 4 - Results
- The final model MSE error is 8.676 and is based on a cross validation preformed with this function cross_val.py
- Class residuals figure example for year 2017:
<p align="center">
    <img src="plots/Final_model_resids_all.png" alt="alternate text">
 </p>
<p align="center">
    <img src="plots/Final_model_resids.png" alt="alternate text">
 </p>

## Part 5 - Application
The application for this model is to help select/dismiss players at the beginning of the season based on their predicted performance.

Use Case Example:
- Using on the model on the 2012 Boston Celtics
<p align="center">
    <img src="Images/Example.png" alt="Boston example">
 </p>

 - This difference, 100 compared to 107 ppg (points per game) can heavily affect the teams position. This can mean a team making the playoffs vs not making the playoffs. Not making the playoffs can mean a loss of millions to the team revenue.
 - Knowing this information prior to the start of the season can give the team a chance to make better business decisions

## Part 6 - Code

The src folder contains 4 py files.
- Coach_Data.py --> This is the code that web scrapes the coach data and saves the data to coach_data_cleaned.csv
- Season Data.py --> This code takes in the original kaggle.com dataset (Season_stats.csv), cleans it and saves it as 'Season_cleaned.csv'
- featurized_season_data.py --> takes in the 'Season_cleaned.csv' dataset, featurizes and saves the data as 'Season_Featurized.csv'
- Cross_Validation.py --> code for running cross validation on the class model
- Model.py --> This is the model. Model takes in the clean featurized season data ('Season_Featurized.csv') and predicts
    - Working with the class model:
        - Creating the class model: model=ModelP('year') # the model is build for predicting for the specified year (works for years 2008 to 2017)
        - Fitting the class model: model.fit(df) ,df=pd.read_csv('data/Season_Featurized.csv', index_col=[0])
        - model.predict() --> returns all the predictions for the required season. model.predict('player name') returns the prediction for the specific player
        - model.score() --> returns the MSE score of the model
        - model.plots('type') --> type = 'hist' or 'residuals' returns the required plot
        - model.clusters() --> returns the predictions for the clusters
        model.importances() --> returns the permutation importance plot of the features
        - model.teams() --> returns the predictions aggregated by team.

## Part 7 - Future Work

Enhance model and model performance:
- Separate players by position
- Apply different weights to each cluster
- Predict other player features such as rebounds, assists et. cetera
Interface:
- Create a user interface
- add current season data

Manual:

- add code example using the class created

## Part 8 - About the author

I'm Eran, data scientist and a product manager. I love technology, and using data to help businesses.

For any questions/comments you can contact me via LinkedIn [Eran-LinkedIn](https://www.linkedin.com/in/eran-shchory-01a717b6)
