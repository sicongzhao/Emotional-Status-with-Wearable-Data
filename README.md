# Emotional-Status-with-Wearable-Data

*I use this repo to demonstrate my data science technical capability to recruiters. To avoid confusion, I have only included code, plots, graphs, and conclusions from my work. For more information, please email sicong.zhao@duke.edu*



### Table of Contents

+ [**Introduction**](#introduction)
+ [**Methods**](#method)
+ [**Data Description**](#data)
+ [**Exploratory Data Analysis**](#EDA)
+ [**Feature Engineering**](#feature)
+ [**Modeling**](#model)
+ [**Evaluation**](#evaluation)




<h2 id="introduction">1.Introduction</h2>

This project examines if emotional states can be reliably recognized from data derived from a commercially available wearable fitness tracker and additional indicators including neuroimaging, personality, and demographic data. 

Currently, we have achieved models for the following scenarios using CatBoost Classifier:

|      | Scenario      | Data                            | Sampling Methods                     | Recall    | Precision | F1        |
| ---- | ------------- | ------------------------------- | ------------------------------------ | --------- | --------- | --------- |
| 1    | Current Users | Fitbit                          | SMOTE                                | 0.550     | 0.347     | 0.425     |
| 2    | New Users     | Fitbit                          | Assign Class Weight to Loss Function | 0.419     | 0.228     | 0.295     |
| 3    | Current Users | All Data (See Data Description) | Assign Class Weight to Loss Function | **0.626** | **0.482** | **0.544** |
| 4    | New Users     | All Data                        | SMOTE                                | 0.870     | 0.260     | 0.399     |

As a comparison, the precision of a random guess is 0.26, the expected F1 score is 0.34.

When predicting emotional states for current subjects, our best two models (#1 and #3) provided a significant overall improvement in these scenarios compared to random guesses. 

The performance for a new user is not ideal, might due to the variation of the emotional trait across people as indicated in exploratory data analysis (EDA). But by using all data we have, the model still does a better than random guessing, improves the expected F1 score by 17.4%.

In terms of **feature importance**, among all our 161 predictors, Neuroticism stands out with the highest feature importance score in terms of both ‘Loss Function Change’ and ‘Prediction Value Change’. It negatively influences human emotion.

![shap_plot](./4-Results/plots/shap_plot.png)


The contributors of this project:
* **Mikella Green**: Data providor, Neuroscience expert
* **Joaquin Menendez**: Data Pre-processing, EDA
* **Sicong Zhao**: EDA, Feature Engineering, Modeling, Data Quality Checking



<h2 id="method">2.Methods</h2>

We followed the traditional data science workflow as shown below. Let me break things down.

![project-flow](./4-Results/report-img/project-flow.png)

**Step1. Pre-processing**

My teammate, Joaquin did most of the work, extracted and organized data points from the original database generated in two experiments from Motivated Cognition and Aging Brain lab in Duke Psychology & Neuroscience department. I checked the data correctness, detected and fixed the scale inconsistency issue for 27 variables.

More details in [**Data Quality Summary**](https://github.com/RyC37/Emotional-Status-with-Wearable-Data/blob/master/1-EDA/%5BReport%5D%20Data%20Quality%20Summary.pdf)

**Step2. EDA** 

I looked into the distributions of emotional states, emotional score distribution by age group, PCA analysis, and KMeans analysis. This step helped us gain a deeper understanding of the data at hand, and resolved the puzzle in terms of which method to use to decide if a subject is happy/unhappy.

More details in [**Exploratory Data Analysis**](#EDA) section.

**Step3. Feature Engineering**

I generated 3 types of features based on others' research and my ideas:

* Heart rate related features
* Steps related features
* Survey related features

More details in [**Feature Engineering**](#feature) section

**Step4. Modeling**

To achieve the best possible precision, I have attacked this problem both in regression and classification approaches. In classification cases, I combined three sampling technologies (upsample minority class, SMOTE, change class weight) to account for the imbalanced dataset due to the positivity bias (people tend to claim they are happy). 
The models I used include CatBoost, XGBoost, Random Forest, Lasso, Ridge, SVM. And CatBoost outperforms than all other models in each scenario.

More detail in [**Modeling**](#model) section.

**Step5. Evaluation**

For regression models, I turned the results into binary results (happy/unhappy). Then evaluated key metrics (F1, recall, precision) using models trained by cross-validation running on a held-out validation set.

More detail in [**Evaluation**](#evaluation) section.



<h2 id="data">3.Data Description</h2>

The data is collected from over 150 participants over 10 days in the Motivated Cognition and Aging Brain lab in Duke Psychology & Neuroscience department, which contains:

| Feature                                                | Frequency                   |
| ------------------------------------------------------ | --------------------------- |
| Measures of personality and behavior                   | 3 times per day per subject |
| Demographic data                                       | one-time measurement        |
| Physical health metrics                                | one-time measurement        |
| Activity tracking data (Steps, Heart Rate form Fitbit) | sum per minute              |
| Functional brain connectivity                          | one-time measurement        |

The data is confidential for now, so I am not including them in this repo. But after our work being published, the data will become public.



<h2 id="eda">4.Exploratory Data Analysis</h2>

In EDA, I scrutinized the following:

* Distribution & Correlation of Labels (Emotional States Measures)
* Relationships between Labels
* Compare the Emotional States by Subjects - check how emotional states vary across subjects
* Valence by Age
* PCA Analysis of Labels
* Emotional States Transformation Analysis
* Emotional States Transformation by Age group

More details in [**Exploratory Data Analysis Report**](https://github.com/RyC37/Emotional-Status-with-Wearable-Data/blob/master/1-EDA/%5BReport%5D%20Exploratory%20Data%20Analysis.pdf)



<h2 id="feature">5.Feature Engineering</h2>

I have created meaningful features from band data (steps & heart rate by minute) within a certain **period** [5m, 10m, 30m, 1h, 3h] (the concept can be shown in the graph below) before the experience sampling (when we record emotional states of participants). 

Features including basic statistics of heart rate and steps, resting time, activity level and variation of heart rate. Among all these features, the ‘variation of heart rate in the last 30 mins’ performs the best. And there are 13 engineered features in the top 30 important features (measured by ‘Loss Function Change’).

![feature-engineering](./4-Results/report-img/feature-engineering.png)

The definition of features generated:

**Steps related features**

- Statistics: max, min, mean, std 
- Move Rate: The # of minutes with step > 0 / total minutes 
- Active Rate: The # of minutes with step > 10 / total minutes 
- Very Active Rate: The # of minutes with step > 20 / total minutes 
- Running Rate: The # of minutes with step > 30 / total minutes

**Heart rate related features**

- Statistics: max, min, mean, std
- Resting Rate: The # of minutes with HR < 30 percentile heart rate > 0 / total minutes
- Moderate Rate: The # of minutes with HR > 50 percentile heart rate > 0 / total minutes
- Very Active Rate: The # of minutes with HR > 80 percentile heart rate > 0 / total minutes
- SDNN: Standard deviation of heartbeat intervals
- pHR2: Percentage of the difference between adjacent HR greater than 2
- rMSSD: Root of mean squared HR change
- Highest HR
- Lowest HR
- l_h: Lowest HR / Highest HR
- CR: Highest HR / Highest HR so far

Code can be found at: [**Feature Engineering Code in Jupyter Notebook**](https://github.com/RyC37/Emotional-Status-with-Wearable-Data/blob/master/Pre-processing-and-feature-engineering.ipynb)



<h2 id="model">6.Modeling</h2>

There are a lot of situations needed to be taken into account. Like the inclusion of features (fitbit only or all features), the scenarios (predict for new users or current users) and different sampling technologies. Therefore, after trying modeling in a lot of [Jupyter notebooks](https://github.com/RyC37/Emotional-Status-with-Wearable-Data/tree/master/3-Model/_Archive), I switched to [a python script](https://github.com/RyC37/Emotional-Status-with-Wearable-Data/blob/master/3-Model/evaluation_tools.py) to control the variables and train models altogether.

We have 1872 observations with 173 features. Given the amount of data we have, the appropriate method types falls into 2 types. Tree based methods and simple models with regularization. 

* For tree based model, we used CatBoost, XGBoost, Random Forest. 
* For simple models with regularization, we used Logistic Regression and Linear Regression with L1, L2 regularization.
* Additionally, we also tried SVM, BaysianRidge as a comparison.

In terms of **hyper parameter tuning**, the initial parameter settings are form my understanding of documentations or others examples. Then I used random grid search and grid search for optimal parameters. 

For **evaluation**, I used cross validation and validate models on held-out set, average the results of F1, recall and precision.

The approaches I have not mentioned before for modeling are as follows:

1. Use random grid search for optimal parameters
2. Use cross validation and validate set to evaluate model performance

The results are as described at [**Introduction**](#introduction).



<h2 id="evaluation">7.Evaluation</h2>

There is [a full list of model evaluation](https://github.com/RyC37/Emotional-Status-with-Wearable-Data/blob/master/4-Results/model_evaluation.csv), which contains all the results I have achieved for now.

In terms of feature importance, here are two plots generated from models use all data for current users. 

Feature importance in the plot below is evaluated by loss function value change.

![Feature-importance-of-new-user-model](./4-Results/plots/Feature-importance-of-new-user-model.png)

As we can see, there are 14 engineered features (4 from step features, 10 from heart rate features) among top 30 important features. That explained why using fitbit alone, we can also achieve a relative good prediction performance for current users.