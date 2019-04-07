# Capstone
In this project I want to predict player performance with relation to the coach.

Data Collection:
- player data collected from kaggle https://www.kaggle.com/drgilermo/nba-players-stats
- coach data was scraped from https://www.basketball-reference.com/coaches/
    - Scraped the data via BeautifulSoup
    - Scraping code is located in the scrape.py file

Data cleaning:
- merged the coach and the player datasets
- removed all data prior to 2005
- removed duplicates and NaN's
- code for data cleaning in cleaning.py

## Part1
Test whether coach effects player performance?
In order to test the relation of the coach to player performance I created a model that uses 4 features and
predicts player's next season ppg (points per game):
- features chosen: player previous season ppg, player previous season minutes played, player age and coach.
- I chose to use a limited amount of player features so it would be easier to identify if the coach effects the prediction

Model 1:
- Created a random forest regressor model that takes in the features discussed above and predicts player ppg
- the benchmark for this model is to predict the previous season ppg. Benchmark MSE=10.36 --> 3.21
- The model predicted with MSE of 8.7 --> which means an average error of 2.9 points in the ppg prediction
- Tested the feature importance of see figure:
<p align="center">
    <img src="plots/model1_FI.png" alt="alternate text">
 </p>
<!-- ![Model1 feature importances](plots/model1_FI.png) -->
<p align="center">
    <img src="plots/model1_resids.png" alt="alternate text">
 </p>
<!-- ![Model1 residuals](plots/model1_resids.png) -->


- This result shows that we get a good prediction only using previous season ppg, age and previous minutes played, whereas
    coach affect isn't showing.

In order to understand this result I tried running the model again after normalizing the data based on player -->
normalized_ppg = player_ppg/max_player_ppg
I did this so players that have a high ppg average don't bias the model.

model2:
- created a random forest model with normalized values
- Used the sam features I used in model 1
- for this model the MSE is 0.044 whereas the benchmark is 0.081
- feature importance is still showing the coach has very limited effect

<p align="center">
    <img src="plots/model2_FI.png" alt="alternate text">
 </p>

 <p align="center">
     <img src="plots/model2_resids.png" alt="alternate text">
  </p>

 Based on this result I decided to further investigate the coach effect:

 model3:
 In this model I tried to measure coach effect but on a team level and not a player level:
 - Tried to predict a team W/L % ratio based on previous season W/L% and coach.
 - Created a random forest model which got a MSE 0.0165 whereas the benchmark got 0.0168

 <p align="center">
     <img src="plots/model3_FI.png" alt="alternate text">
 </p>
 <p align="center">
      <img src="plots/model3_resids.png" alt="alternate text">
 </p>

 <p align="center">
       <img src="plots/model3_bar.png" alt="alternate text">
 </p>

 - again the result show that the significance of the coach is low

 Based on these results I decided to set up an hypothesis test that will determine the significance level of the coach.

## part2
Hypothesis testing:
- Null Hypothesis --> H<sub>0</sub> - coach has no effect --> W/L% stays the same after team changes a coach
- Alternative Hypothesis --> H<sub>a</sub> - team W/L% changes when a new coach comes
- Significance level alpha = 5%
- Ran the test comparing between W/L ratio between a season after a coach change and before

Assumptions:
- One game is a Bernoulli trial --> G ~ Bernoulli(P), E[G]=P
- One season of N games is a Binomial distribution --> S ~ Bin(N,P), E[S]=N*P
- S
- x&#772;
- S&#772;
