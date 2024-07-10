## Arda Gediz
## PL Predictor using scikit-learn to predict from the matches.csv stat sheet containing data from all matches from 2022-2020

import pandas as pd 
matches = pd.read_csv("matches.csv", index_col = 0)

##converting all objects to int or float to be processed by the machine learning software
matches["date"] = pd.to_datetime(matches["date"])
matches["h/a"] = matches["venue"].astype("category").cat.codes ## converting venue to a home (1) or away (0) number
matches["opp"] = matches["opponent"].astype("category").cat.codes ## converting opponents to a number
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int") ## converting hours to number in case a team plays better at a certain time
matches["day"] = matches["date"].dt.dayofweek ## converting day of week of game to a number