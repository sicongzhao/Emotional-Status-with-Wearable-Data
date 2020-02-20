# Emotional-Status-with-Wearable-Data

### 1.Overview
This is an ongoing project about determining if negative emotional states can be predicted from wearable activity sensor data.
Currently, the major achievements are:

1. A classification model with:
* AUC-RoC: 0.76
* Precision: 0.61
* Recall: 0.31
* F1-Score: 0.40
As a comparison, the precision of random guess is 0.11.

2. Insights negative emotion: Among all our 161 predictors, Neuroticism stands out with highest feature importance score in terms of both ‘Loss Function Change’ and ‘Prediction Value Change’. It negatively influence human emotion.

### 2.Data Description
In this project, I trained models based on data collected from over 150 participants over 10 days in the Motivated Cognition and Aging Brain lab in Duke Psychology & Neuroscience department, which contains:

* measures of personality and behavior
* demographic data
* physical health metrics
* activity tracking data (Fitbit)
* functional brain connectivity

### 3.Feature Engineering
I have created meaningful features from band data (steps & heart rate by minute) within a certain period [5m, 10m, 30m, 1h, 3h] before the experience sampling (when we record emotional states of participants). Features including basic statistics of hear rate and steps, resting time, activity level and variation of heart rate. Among all these features, the ‘variation of heart rate in last 30 mins’ performs the best. And there are 13 engineered feature in top 30 important features (measured by ‘Loss Function Change’).

### 4.Modeling & Analysis
From a business point of view, we need to account for new users and existing users. So I ended up with 2 type models. With first type trained on stratified split data and second type trained on data split by group. Due to the positivity bias (people tend to say they are happy), we have imbalanced data with only 10.8% records report negative emotion. To account for that, I tested different sample technologies (downsample majority class, oversample minority class, SMOTE) with different type of models. The best performance model is trained through oversample minority class with CatBoost.
