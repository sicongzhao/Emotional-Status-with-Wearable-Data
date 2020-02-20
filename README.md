# Emotional-Status-with-Wearable-Data
This is an ongoing project about determining if negative emotional states can be predicted from wearable activity sensor data.
Currently, the major achievements are:

1. A classification model with:
* AUC-RoC: 0.76
* Precision: 0.63
* Recall: 0.31
* F1-Score: 0.40
As a comparison, the precision of random guess is 0.11.

2. Insights negative emotion: Among all our 161 predictors, Neuroticism stands out with highest feature importance score in terms of both ‘Loss Function Change’ and ‘Prediction Value Change’. It negatively influence human emotion.

In order to achieve our goal, we trained our models based on data collected from the Motivated Cognition and Aging Brain lab in Duke Psychology & Neuroscience department, which contains:

· measures of personality and behavior
· demographic data
· physical health metrics
· activity tracking data (Fitbit)
· functional brain connectivity
