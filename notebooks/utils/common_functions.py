#!/usr/bin/env python
# coding: utf-8

# 
# ## Utility Functions used in Notebooks
# 

# In[4]:


from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, make_scorer, average_precision_score, precision_recall_curve, roc_curve, f1_score, roc_auc_score, ConfusionMatrixDisplay,make_scorer, RocCurveDisplay, PrecisionRecallDisplay

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, RFE
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


# In[5]:


####################################################################################################################################################
# Generic Function to cap outliers in a specific column of input dataframe to the given lower and upper bounds 
####################################################################################################################################################
def cap_outliers(df_cleaned, column, lower_bound, upper_bound):
    """
    Cap outliers in a specified column of input DataFrame to the given bounds.

    Parameters:
    - df: pandas DataFrame
    - column: str, name of the column to process
    - lower_bound: float, lower bound for capping
    - upper_bound: float, upper bound for capping

    Returns:
    - DataFrame with outliers capped
    """
    print(f"\nColumn {column} has outliers greater than upper bound ({upper_bound}) or lower than lower bound ({lower_bound}). Capping them now.")
    df_cleaned[column] = df_cleaned[column].clip(lower=lower_bound, upper=upper_bound)
    return df_cleaned


# In[6]:


####################################################################################################################################################
# Generic Function to identify, display and drop duplicate rows in input dataframe
####################################################################################################################################################
def handle_duplicates(df):
    """
    Identifies, displays, counts, and drops duplicate rows in a DataFrame.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - DataFrame with duplicate rows removed
    """
    # Identify duplicate rows
    duplicate_rows = df.duplicated()
    
    # Display duplicate rows
    duplicates = df[duplicate_rows]
    print("\nActual Duplicate Rows:")
    print(duplicates)
    
    # Count duplicate rows
    duplicate_count = duplicate_rows.sum()
    print("\nNumber of Duplicate Rows:", duplicate_count)
    
    # Drop duplicate rows
    df_cleaned = df.drop_duplicates()
    print("\nAfter dropping duplicate rows, the count of duplicate rows now: ", df_cleaned.duplicated().sum())
    
    return df_cleaned


# In[7]:


###################################################################################################
# Generic Model Evaluation function to get additional metrics used across all classifiers
###################################################################################################

# Generic Function to get additional metrics given an estimator, test feature set and target labels
def evaluate_model(estimator, Xt_test, yt_test, threshold=0.5):
    """
    Evaluates a fitted decision tree pipeline using various performance metrics, including
    Precision-Recall, ROC AUC curves, and confusion matrix, and displays plots.

    Parameters:
    estimator (Model): The fitted model.
    Xt_test (pd.DataFrame): The test feature set.
    yt_test (pd.Series): The test target labels.
    threshold: Threshold.
    Returns:
    dict: A dictionary with various evaluation metrics and confusion matrix details.
    """
    # Predict class labels for multiclass prediction
    y_pred = estimator.predict(Xt_test)

    # Predict probabilities for multiclass classification
    y_pred_proba = estimator.predict_proba(Xt_test)

    # Compute F1 score on test set
    test_f1 = f1_score(yt_test, y_pred, average='macro')
    test_f1_weighted = f1_score(yt_test, y_pred, average='weighted')

    # Compute Accuracy score on test set
    test_accuracy = accuracy_score(yt_test, y_pred)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(yt_test, y_pred))

    # Compute PR AUC and ROC AUC using one-vs-rest for each class
    pr_auc_values = []
    roc_auc_values = []

    for cls in np.unique(yt_test):
        # Convert to binary: 1 if the class is `cls`, 0 otherwise
        y_test_binary = (yt_test == cls).astype(int)

        # Get class-specific predicted scores (log-probabilities for PR AUC and ROC AUC)
        y_score_proba = y_pred_proba[:, cls]

        # Precision-Recall AUC for class `cls`
        pr_auc_value = average_precision_score(y_test_binary, y_score_proba)
        pr_auc_values.append(pr_auc_value)
        print(f"PR AUC for class {cls}: {pr_auc_value:.2f}")

        # ROC AUC for class `cls`
        fpr, tpr, _ = roc_curve(y_test_binary, y_score_proba)
        roc_auc_value = auc(fpr, tpr)
        roc_auc_values.append(roc_auc_value)
        print(f"ROC AUC for class {cls}: {roc_auc_value:.2f}")

    # Macro-averaged PR AUC (treat each class equally)
    pr_auc_macro = np.mean(pr_auc_values)
    print(f"Macro-averaged PR AUC: {pr_auc_macro:.2f}")

    # Macro-averaged ROC AUC
    roc_auc_macro = np.mean(roc_auc_values)
    print(f"Macro-averaged ROC AUC: {roc_auc_macro:.2f}")

    # Compute the weighted average PR AUC score
    class_counts = np.bincount(yt_test)
    pr_auc_weighted = np.average(pr_auc_values, weights=class_counts)
    print(f"Weighted PR AUC Score: {pr_auc_weighted:.2f}")

    # Compute roc_auc_score with 'weighted' averaging
    roc_auc_weighted = roc_auc_score(yt_test, y_pred_proba, average='weighted', multi_class='ovr')
    print(f"Weighted ROC AUC Score: {roc_auc_weighted:.2f}")

    # Plot subplots
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot1. Confusion Matrix for Multiclass Classification
    conf_matrix = confusion_matrix(yt_test, y_pred)
    disp1 = ConfusionMatrixDisplay(conf_matrix)
    disp1.plot(ax=ax[0])
    ax[0].set_title('Confusion Matrix')

    # Plot2. Precision-Recall Curves for each class (One-vs-Rest)
    for cls in np.unique(yt_test):
        # Binarize the true labels and predictions for the current class
        yt_test_binary = (yt_test == cls).astype(int)
        y_pred_proba_cls = y_pred_proba[:, cls]

        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(yt_test_binary, y_pred_proba_cls)

        # Plot Precision-Recall curve
        disp2 = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp2.plot(ax=ax[1], name=f'Class {cls} (PR AUC = {pr_auc_values[cls]:.2f})')

    ax[1].set_title('Precision-Recall Curves (One-vs-Rest)')

    # Plot3. ROC Curves for each class (One-vs-Rest)
    for cls in np.unique(yt_test):
        # Binarize the true labels and predictions for the current class
        yt_test_binary = (yt_test == cls).astype(int)
        y_pred_proba_cls = y_pred_proba[:, cls]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(yt_test_binary, y_pred_proba_cls)

        # Plot ROC curve
        disp3 = RocCurveDisplay(fpr=fpr, tpr=tpr)
        disp3.plot(ax=ax[2], name=f'Class {cls} (ROC AUC = {roc_auc_values[cls]:.2f})')

    ax[2].set_title('ROC AUC Curves (One-vs-Rest)')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # Return evaluation metrics in a dictionary
    results = {
        'f1_score_macro'   : test_f1,
        'f1_score_weighted': test_f1_weighted,
        'accuracy_score'  : test_accuracy,
        'pr_auc_macro'    : pr_auc_macro,
        'roc_auc_macro'   : roc_auc_macro,
        'pr_auc_weighted' : pr_auc_weighted,
        'roc_auc_weighted': roc_auc_weighted,
        'confusion_matrix': conf_matrix
    }
    return results


# In[8]:


###################################################################################################
# Generic Function to analyze Feature Selection results from input Pipeline 
###################################################################################################
def analyze_feature_selection(pipeline):
    """
    Analyze feature selection results from a pipeline.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): A fitted pipeline containing a feature selection step.

    Returns:
    tuple: A tuple containing two elements:
           1. A list of tuples (feature_name, rank) for all features, sorted by rank.
           2. A list of selected feature names.
    """
    def get_feature_names(column_transformer):
        feature_names = []
        for name, transformer, columns in column_transformer.transformers_:
            if name == 'passthrough':
                feature_names.extend(columns)
            elif hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(input_features=columns))
            else:
                feature_names.extend(columns)
        return feature_names

    # Get feature names from the preprocessor
    feature_names = get_feature_names(pipeline.named_steps['preprocessor'])

    # Get feature selector
    feature_selector = pipeline.named_steps['feature_selection']

    # Get selected indices
    selected_indices = feature_selector.get_support(indices=True)

    # Initialize feature ranking
    feature_ranking = []
    # Get feature ranking if available
    if hasattr(feature_selector, 'ranking_'):
        ranking = feature_selector.ranking_
        feature_ranking = list(zip(feature_names, ranking))
        feature_ranking.sort(key=lambda x: x[1])

    # Verify the length of selected_indices and feature_names
    if max(selected_indices) >= len(feature_names):
        raise ValueError(f"Error: Selected indices {selected_indices} exceed the number of features ({len(feature_names)}).")

    # Map selected indices to feature names
    selected_feature_names = [feature_names[i] for i in selected_indices]

    return feature_ranking, selected_feature_names


# In[9]:


###################################################################################################
# Generic Function to plot feature importances from a trained classifier
###################################################################################################
def plot_feature_importances(classifier, selected_features):
    """
    Plots feature importances from a trained classifier.

    Parameters:
    - classifier: Trained model (e.g., DecisionTreeClassifier, RandomForestClassifier, LogisticRegressionClassifier).
    - selected_features: List of RFE selected feature names corresponding to the model's input features.
    """
    # Get the feature importances from the classifier
    feature_importances = classifier.feature_importances_

    # Create a dictionary of feature names and their corresponding importance values
    importance_dict = dict(zip(selected_features, feature_importances))

    # Sort the features by their importance values in descending order
    sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Print the sorted feature importances
    print("\nFeature Importances (sorted):")
    for feature, importance in sorted_importances:
        print(f"{feature}: {importance:.4f}")

    # Plotting feature importances
    plt.figure(figsize=(10, 6))

    # Unpack the sorted features and their importance values
    features, importances = zip(*sorted_importances)

    # Create a bar plot to visualize the feature importances
    plt.bar(features, importances, color='skyblue')
    plt.xticks(rotation=90)
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')

    # Adjust the layout to prevent clipping of labels
    plt.tight_layout()
    plt.show()


# In[ ]:




