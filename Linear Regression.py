# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:16:24 2020

@author: Antoine, Yunhan
"""

#-----------------------------------REGRESSION---------------------------------

import pandas as pd
import seaborn as sns

#read the data
data = pd.read_csv("moviemetadata_python.csv")

data["gross_inflation_adjusted_2019"]=data["gross_inflation_adjusted_2019"]/1000000
data["budget_inflation_adjusted_2019"]=data["budget_inflation_adjusted_2019"]/1000000

#Exploring the data
data.head() 

#List of columns
data.columns

sns.pairplot(data)

#Using the correlation method, we can assess how correlated variables are to one another
#The correlation coefficient is an index that ranges from -1 to 1. 
#A value of 0 means that there is no correlation while a value closer to 1 or -1 means positive or negative correlation
correlation_values=data.corr()

sns.heatmap(correlation_values)

#Preprocessing the data
##create a temporary variable that contains columns main_genres and color
temp = data[["main_genres","color"]]

#Convert to dummy variables
temp = pd.get_dummies(temp,prefix=["main_genres","color"])

#dropping the redundant variables
temp = temp.drop(["color_Black and White"], axis =1)

#train is the preprocessed data that we will use
##only variables that are known to the public prior to the movie release were kept 
temp2 = data.drop(["genres", "plot_keywords","language","country",
                 "content_rating","color","aspect_ratio","budget","gross",
                 "imdb_score", "num_user_for_reviews","num_critic_for_reviews",
                 "num_voted_users","movie_facebook_likes","director_name",
                 "director_facebook_likes","actor_1_name", "actor_1_facebook_likes",
                 "actor_2_name", "actor_2_facebook_likes","actor_3_name", 
                 "actor_3_facebook_likes","cast_total_facebook_likes", "movie_imdb_link",
                 "movie_title","main_genres"], axis =1)

train = pd.concat([temp,temp2], axis =1)

#Creating independent variable and dependent variable
y = train["gross_inflation_adjusted_2019"]

x = train.drop(["gross_inflation_adjusted_2019"], axis=1)

from sklearn.model_selection import train_test_split

#Splits the data into training and testing data
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test.shape)

#Import regression model
from sklearn.linear_model import LinearRegression 

#Initialize the model
lm = LinearRegression()

lm.fit(x_train, y_train) #Training the model

print (lm.intercept_)
print (lm.coef_)
print (lm.coef_.tolist())

x.columns

predictions = lm.predict(x_test)

import matplotlib.pyplot as plt
plt.scatter(predictions, y_test)

from sklearn.metrics import r2_score

r2_score(y_test, predictions)









