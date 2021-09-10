# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:38:52 2020

@author: Antoine
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset=pd.read_csv('moviemetadata_python.csv')
dataset.head()

dataset = dataset.drop(["movie_title","main_genres","genres","plot_keywords", "language",
              "country", "color", "movie_imdb_link", "facenumber_in_poster", 
              "director_name", "actor_1_name", "actor_2_name", "actor_3_name",
              "content_rating","aspect_ratio"], axis=1)

dataset.describe()

#visualize
sns.pairplot(dataset)

#using the elbow method to find the ideal number of clusters
from sklearn.cluster import KMeans 

wcv = []#created empty list

for i in range(1, 11):
    km=KMeans(n_clusters=i)
    km.fit(dataset)
    wcv.append(km.inertia_)
                             
plt.plot(range(1, 11), wcv)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('wcv')
plt.show()

#Fitting kmeans to the dataset
km4=KMeans(n_clusters=4, random_state=0) #Initialize

y_means = km4.fit_predict(dataset) #train

dataset["LABEL"]=y_means


sns.scatterplot(x="imdb_score", y="gross_inflation_adjusted_2019", hue="LABEL", palette="Set1", data=dataset)
sns.scatterplot(x="actor_1_facebook_likes", y="gross_inflation_adjusted_2019", hue="LABEL", palette="Set1", data=dataset)
sns.scatterplot(x="num_user_for_reviews", y="gross_inflation_adjusted_2019", hue="LABEL", palette="Set1", data=dataset)

