# Emotional-Status-with-Wearable-Data

### Table of Contents
+ [**Introduction**](#introduction)
+ [**Data Description**](#data)
+ [**Exploratory Data Analysis**](#EDA)
+ [**Feature Engineering**](#feature)
+ [**Modeling**](#model)


<h2 id="introduction">1.Introduction</h2>

As the popularity and capability of wearable activity trackers has increased over the last 20 years, so has an interest in what data derived from their internal sensors can tell us about our overall health. 

This project examines if emotional states can be reliably recognized from data derived from a commercially available wearable fitness tracker and additional indicators including neuroimaging, personality, and demographic data. 

Currently, we have achieved:

1. A classification model with:
* AUC-RoC: 0.76
* Precision: 0.61
* Recall: 0.31
* F1-Score: 0.40
As a comparison, the precision of random guess is 0.11.

2. Insights negative emotion: Among all our 161 predictors, Neuroticism stands out with highest feature importance score in terms of both ‘Loss Function Change’ and ‘Prediction Value Change’. It negatively influence human emotion.


The contributors of this project:
* **Mikella Green**: Data providor, Neuroscience expert
* **Joaquin Menendez**: Data Pre-processing, Modeling
* **Sicong Zhao**: Feature Engineering, Modeling, Data Quality Checking

*Since this is a personal repo for recruiting purpose, I am only including code and conclusion from my work. For more information, please email sicong.zhao@duke.edu*

<h2 id="data">2.Data Description</h2>
The data is collected from over 150 participants over 10 days in the Motivated Cognition and Aging Brain lab in Duke Psychology & Neuroscience department, which contains:

* measures of personality and behavior
* demographic data
* physical health metrics
* activity tracking data (Fitbit)
* functional brain connectivity

<h2 id="eda">3.Exploratory Data Analysis</h2>

In EDA, I scrutinized following:
* Distribution & Correlation of Labels (Emotional States Measures)
* Relationships bewteen Labels
* Compare Emotional States by Subjects - check how emotional states vary across subjects
* Valence by Age
* PCA Analysis of Labels
* Emotional States Transformation Analysis
* Emotional States Transformation by Age group

<h2 id="feature">4.Feature Engineering</h2>
I have created meaningful features from band data (steps & heart rate by minute) within a certain period [5m, 10m, 30m, 1h, 3h] before the experience sampling (when we record emotional states of participants). Features including basic statistics of hear rate and steps, resting time, activity level and variation of heart rate. Among all these features, the ‘variation of heart rate in last 30 mins’ performs the best. And there are 13 engineered feature in top 30 important features (measured by ‘Loss Function Change’).
* [Feature Engineering Code in Jupyter Notebook](https://github.com/RyC37/Emotional-Status-with-Wearable-Data/blob/master/Pre-processing-and-feature-engineering.ipynb)


<h2 id="model">5.Modeling</h2>
From a business point of view, we need to account for new users and existing users. So I ended up with 2 type models. With first type trained on stratified split data and second type trained on data split by group. Due to the positivity bias (people tend to say they are happy), we have imbalanced data with only 10.8% records report negative emotion. To account for that, I tested different sample technologies (downsample majority class, oversample minority class, SMOTE) with different type of models. The best performance model is trained through oversample minority class with CatBoost.
* [Predict Emotion for Current Users](https://github.com/RyC37/Emotional-Status-with-Wearable-Data/blob/master/Predict_Positive_or_Negative_Emotion_for_Current_User.ipynb)
* [Predict Emotion for New Users](https://github.com/RyC37/Emotional-Status-with-Wearable-Data/blob/master/Predict_Positive_or_Negative_Emotion_for_New_User.ipynb)

I have also tried to tackle this problem as a regression problem, the result is not ideal and I have relative high RMSE, but I would like to share the result:
* [Predict Emotion Score for Current Users](https://github.com/RyC37/Emotional-Status-with-Wearable-Data/blob/master/Predict_Valence_Score_for_Current_Users.ipynb)
