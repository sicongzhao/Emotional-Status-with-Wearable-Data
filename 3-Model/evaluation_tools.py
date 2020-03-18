#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, \
    recall_score, precision_score, f1_score, auc, roc_curve, \
    precision_recall_curve, r2_score, mean_squared_error
import seaborn as sns
import catboost
import shap

def eval_class(y_test, y_pred):
    '''
  This function evaluates the model performance
  in terms of Accuracy, Recall, Precision, Specificity and F1
  
  Parameters:
  y_test (array):     Test label
  y_pred (array):     Model prediction

  Returns:
  Evaluation information printed
  array contains [accuracy, recall, precision, specificity, npv, f1]
  '''

  # Accuracy

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    (tn, fp, fn, tp) = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    f1 = f1_score(y_test, y_pred)

#     print ('Accuracy: \t', str(accuracy))
#     print ('Recall: \t', str(recall))
#     print ('Precision: \t', str(precision))
#     print ('Specificity: \t', str(specificity))
#     print ('NPV: \t', str(npv))
#     print ('F1: \t\t', str(f1))
    return [
        accuracy,
        recall,
        precision,
        specificity,
        npv,
        f1
        ]


def plot_roc(y_test, prob, model_type):
    '''
  This function plots RoC curve for classfication algotrithms.
  
  Parameters:
  y_test (array):       Test label
  prob (array):         Model prediction of probability for binary class
  model_type (string):  Model names

  Returns: 
  RoC plot
  '''

    preds = prob[:, 1]
    (fpr, tpr, threshold) = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        preds = prob[:, 0]
        (fpr, tpr, threshold) = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)

  # Plot

    plt.figure(figsize=(5.5, 4))
    plt.title(model_type + ' RoC')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def confision_matrix(y_test, y_pred, model_type):
    '''
  This function plots confision matrix for classfication algotrithms.
  
  Parameters:
  y_test (array):     Test label
  y_pred (array):     Model prediction
  model_type (string):  Model names

  Returns: 
  Confusion Matrix plot
  '''

    cm_2 = confusion_matrix(y_test, y_pred)
    cm_2_df = pd.DataFrame(cm_2)
    plt.figure(figsize=(5.5, 4))
    ax = sns.heatmap(cm_2_df, annot=True)
    ax.set_ylim(0, 2)
    accuracy = accuracy_score(y_test, y_pred)
    plt.title(model_type
              + ' Confusion Matrix \nAccuracy:{0:.3f}'.format(accuracy))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_pr_curve(
    y_test,
    prob,
    y_pred,
    model_type,
    ):
    '''
  This function plots PR curve for classfication algotrithms.
  
  Parameters:
  y_test (array):       Test label
  prob (array):         Model prediction of probability for binary class
  y_pred (array):     Model prediction
  model_type (string):  Model names

  Returns: 
  PR plot
  '''

    (lr_precision, lr_recall, _) = precision_recall_curve(y_test, prob[:
            , 1])
    (lr_f1, lr_auc) = (f1_score(y_test, y_pred), auc(lr_recall,
                       lr_precision))

  # summarize scores

    print (model_type, ' PR Curve: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))

  # plot the precision-recall curves

    unhappy = len(y_test[y_test == 1]) / len(y_test)
    plt.figure(figsize=(5.5, 4))
    plt.plot([0, 1], [unhappy, unhappy], linestyle='--', label='Unhappy'
             )
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')

  # axis labels

    plt.xlabel('Recall')
    plt.ylabel('Precision')

  # show the legend

    plt.legend()

  # show the plot

    plt.show()


def automate(
    y_test,
    prob,
    y_pred,
    model_type,
    ):
    plot_roc(y_test, prob, model_type)
    confision_matrix(y_test, y_pred, model_type)
    plot_pr_curve(y_test, prob, y_pred, model_type)
    eval_class(y_test, y_pred)


# Functions for feature evaluation

def log_loss(m, X, y):
    return metrics.log_loss(y, m.predict_proba(X)[:, 1])


def mse(model, y_test, X_test):
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)


def permutation_importances(
    model,
    X,
    y,
    metric,
    ):
    baseline = metric(model, X, y)
    imp = []
    for col in X.columns:
        save = X[col].copy()
        X[col] = np.random.permutation(X[col])
        m = metric(model, X, y)
        X[col] = save
        imp.append(m - baseline)
    return np.array(imp)


def baseline_importance(
    model,
    X,
    y,
    X_test,
    y_test,
    metric,
    ):

    model = CatBoostClassifier(one_hot_max_size=10, iterations=500)
    model.fit(X, y, cat_features=categorical_features_indices,
              verbose=False)
    baseline = metric(model, X_test, y_test)

    imp = []
    for col in X.columns:

        save = X[col].copy()
        X[col] = np.random.permutation(X[col])

        model.fit(X, y, cat_features=categorical_features_indices,
                  verbose=False)
        m = metric(model, X_test, y_test)
        X[col] = save
        imp.append(m - baseline)
    return np.array(imp)


def get_feature_imp_plot(
    model,
    method,
    X_train,
    y_train,
    X_test,
    y_test,
	categorical_features_indices
    ):

    if method == 'Permutation':
        fi = permutation_importances(model, X_test, y_test, log_loss)
    elif method == 'Baseline':

        fi = baseline_importance(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            log_loss,
            )
    elif method == 'ShapValues':

        shap_values = \
            model.get_feature_importance(catboost.Pool(X_test,
                label=y_test,
                cat_features=categorical_features_indices),
                type='ShapValues')
        shap_values = shap_values[:, :-1]
        shap.summary_plot(shap_values, X_test)
    elif method == 'ShapValues_class':

        shap_values = \
            model.get_feature_importance(catboost.Pool(X_test,
                label=y_test,
                cat_features=categorical_features_indices),
                type='ShapValues')

        # SHAP value for class 1

        shap_values = shap_values[:, :, :-1]
        shap.summary_plot(shap_values[:, 1], X_test)
    else:

        fi = model.get_feature_importance(catboost.Pool(X_test,
                label=y_test,
                cat_features=categorical_features_indices), type=method)

    if method != 'ShapValues' and method != 'ShapValues_class':
        feature_score = pd.DataFrame(list(zip(X_test.dtypes.index,
                fi)), columns=['Feature', 'Score'])

        feature_score = feature_score.sort_values(by='Score',
                ascending=False, inplace=False, kind='quicksort',
                na_position='last')

        plt.rcParams['figure.figsize'] = (60, 7)
        ax = feature_score.plot('Feature', 'Score', kind='bar',
                                color='c')
        ax.set_title('Feature Importance using {}'.format(method),
                     fontsize=14)
        ax.set_xlabel('features')
        plt.show()

def convert_reg_to_class(y_regr):
	return [0 if x >= 0 else 1 for x in y_regr]

def eval_reg(y_test, y_pred):
	'''
	This function evaluates the regression model performance
	in terms of RMSE, R2
  
	Parameters:
	y_test (array):     Test label
	y_pred (array):     Model prediction

	Returns:
	Evaluation information printed
	array contains [RMSE, r2]
	'''
	RMSE = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	print('RMSE: \t', str(RMSE))
	print('r2: \t', str(r2))
	return [RMSE, r2]


def catboost_feature_importance(model, feature_col):
	# Extract feature importance
	feature_importance = pd.Series(model.feature_importances_)
	feature_name = pd.Series(feature_col)
	feature_importance_score = pd.DataFrame(dict(feature_name = feature_name, score = feature_importance))
	# Sort by score
	return feature_importance_score.sort_values(by='score', ascending=False)
	