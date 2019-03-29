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

Part1 - Hypothesis testing
Test wether coach effects player performance?
In order to test the relation of the coach to player performance I created a model that uses 4 features and
predicts player's next season ppg (points per game):
- features chosen: player previous season ppg, player previous season minutes played, player age and coach.
- I chose to use a limited amount of player features so it would be easier to identify if the coach effects the prediction

Model 1:
- Created a random forest regressor model that takes in the features discussed above and predicts player ppg
- the benchmark for this model is to predict the previous season ppg. Benchmark MSE=10.36 --> 3.21
- The model predicted with MSE of 8.7 --> which means an average error of 2.9 points in the ppg prediction
- Tested the feature importance of see figure:
![Model1 feature importances](plots/model1_FI.png)
<p align="center">
    <img src="plots/model1_resids.png" alt="alternate text">
 </p>
![Model1 residuals](plots/model1_resids.png)


- This result shows that we get a good prediction only using previous season ppg, age and previous minutes played, whereas
    coach affect isn't showing.

In order to understand this result I tried running the model again after normalizing the data based on player -->
normalized_ppg = player_ppg/max_player_ppg
I did this so players that have a high ppg average don't bias the model.

model2:
- created a random forest model with normalized values
-
