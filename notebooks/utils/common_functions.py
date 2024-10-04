#!/usr/bin/env python
# coding: utf-8

# 
# ## Utility Functions used in Notebooks
# 

# In[5]:


from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, make_scorer, average_precision_score, precision_recall_curve, roc_curve, f1_score, roc_auc_score, ConfusionMatrixDisplay, make_scorer, RocCurveDisplay, PrecisionRecallDisplay

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, RFE
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


# In[2]:


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


# In[ ]:


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


# In[ ]:


###################################################################################################
# Define a custom F2 scorer with average=weighted
###################################################################################################
from sklearn.metrics import fbeta_score

def f2_weighted(y_true, y_pred, beta=2):
    """
    Calculate the weighted F2 score for multi-class classification.

    The F2 score is a measure of a model's accuracy that combines precision and recall,
    with beta=2 giving more weight to recall than precision.

    Parameters:
        y_true (Series): Actual target values.
        y_pred (Series): Estimated targets as returned by a classifier.

    Returns:
        float: The weighted F2 score. Best value is 1.0 and worst value is 0.0.

    Notes:
        - This function uses scikit-learn's fbeta_score with beta=2.
        - The 'weighted' average calculates metrics for each label, and finds their average
      weighted by support (the number of true instances for each label).
    """
    return fbeta_score(y_true, y_pred, beta=2, average='weighted')


# In[ ]:


###################################################################################################
# Define a custom F2 scorer with more weights to minority classes
###################################################################################################
from sklearn.metrics import fbeta_score

def f2_with_custom_weights(y_true, y_pred, beta=2):
    """
    Compute a custom-weighted F-beta score for multi-class classification.

    This function calculates the F-beta score for each class separately and then
    computes a weighted average, giving more importance to minority classes.

    Parameters:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        beta (float): The beta parameter for the F-beta score. Default is 2 for F2 score.

    Returns:
        float: Weighted average of F-beta scores across all classes
    """
    # Compute F-beta score for each class
    f_scores = fbeta_score(y_true, y_pred, beta=beta, average=None)

    # Weigh the scores with more weight to minority classes
    weights = [1, 2, 3]

    return np.average(f_scores, weights=weights)


# In[ ]:


###################################################################################################
# Generic Model Evaluation function to get additional metrics used across all classifiers
###################################################################################################
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Generic Function to get additional metrics given an estimator, test feature set and target labels
def evaluate_model(estimator, Xt_test, yt_test, X_val, y_val, threshold=0.5):
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
    # Evaluate on validation set
    val_pred = estimator.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    val_accuracy = accuracy_score(y_val, val_pred)

    print(f"Validation F1 Score: {val_f1}")
    print(f"Validation Accuracy: {val_accuracy}")

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
        'val_f1_score'      : val_f1,
        'val_accuracy_score': val_accuracy,
        'f1_score_macro'    : test_f1,
        'f1_score_weighted' : test_f1_weighted,
        'accuracy_score'  : test_accuracy,
        'pr_auc_macro'    : pr_auc_macro,
        'roc_auc_macro'   : roc_auc_macro,
        'pr_auc_weighted' : pr_auc_weighted,
        'roc_auc_weighted': roc_auc_weighted,
        'confusion_matrix': conf_matrix
    }
    return results


# In[6]:


###################################################################################################
# Generic Model Evaluation function with adjusted threshold for class2. Used to get additional
# metrics used across all classifiers.
###################################################################################################
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model_with_adjusted_threshold_for_class2(estimator, Xt_test, yt_test, X_val, y_val):
    """
    Evaluates a fitted model using various performance metrics and adjusts the threshold for class 2
    based on F1 score optimization.

    Parameters:
        estimator (Model): The fitted model.
        Xt_test (pd.DataFrame): The test feature set.
        yt_test (pd.Series): The test target labels.
        X_val (pd.DataFrame): The validation feature set.
        y_val (pd.Series): The validation target labels.

    Returns:
        dict: A dictionary with various evaluation metrics and confusion matrix details.
    """
    # Evaluate on validation set
    val_pred = estimator.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    val_accuracy = accuracy_score(y_val, val_pred)

    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Predict probabilities for multiclass classification
    y_pred_proba = estimator.predict_proba(Xt_test)

    # Calculate original predictions
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Adjust the threshold for class 2 based on F1 score optimization
    precision, recall, thresholds = precision_recall_curve(yt_test == 2, y_pred_proba[:, 2])
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]

    # Adjust predictions for class 2 using the best threshold
    y_pred_adjusted = np.copy(y_pred)
    y_pred_adjusted[y_pred_proba[:, 2] >= best_threshold] = 2

    # Compute F1 score on test set with adjusted predictions
    test_f1 = f1_score(yt_test, y_pred_adjusted, average='macro')
    test_f1_weighted = f1_score(yt_test, y_pred_adjusted, average='weighted')

    # Compute Accuracy score on test set
    test_accuracy = accuracy_score(yt_test, y_pred_adjusted)

    # Calculate and display F1 scores for each class
    f1_per_class = f1_score(yt_test, y_pred_adjusted, average=None)

    # Display F1 scores for each class
    class_labels = np.unique(yt_test)
    print("\nF1 Scores per Class:")
    for cls, f1_score_temp in zip(class_labels, f1_per_class):
        print(f"Class {cls}: F1 Score = {f1_score_temp:.4f}")

    # Display macro and weighted F1 scores
    print(f"\nMacro-Averaged F1 Score: {test_f1:.4f}")
    print(f"Weighted-Averaged F1 Score: {test_f1_weighted:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")

    # Classification report
    print("\nClassification Report with Adjusted Threshold for class2 based on F1 score optimization:")
    print(classification_report(yt_test, y_pred_adjusted))

    # Compute PR AUC and ROC AUC using one-vs-rest for each class
    pr_auc_values = []
    roc_auc_values = []

    for cls in np.unique(yt_test):
        # Convert to binary: 1 if the class is `cls`, 0 otherwise
        y_test_binary = (yt_test == cls).astype(int)

        # Get class-specific predicted scores (probabilities for PR AUC and ROC AUC)
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
    conf_matrix = confusion_matrix(yt_test, y_pred_adjusted)
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
        'val_f1_score': val_f1,
        'val_accuracy_score': val_accuracy,
        'f1_score_macro': test_f1,
        'f1_score_weighted': test_f1_weighted,
        'accuracy_score': test_accuracy,
        'pr_auc_macro': pr_auc_macro,
        'roc_auc_macro': roc_auc_macro,
        'pr_auc_weighted': pr_auc_weighted,
        'roc_auc_weighted': roc_auc_weighted,
        'confusion_matrix': conf_matrix,
        'best_threshold': best_threshold
    }

    return results


# In[ ]:


##################################################################################################################
# Generic evaluation function that focusses more on recall specifically for minority classes - class 2.
# This modification over existing functions evaluate_model, evaluate_model_with_adjusted_threshold_for_class2,
# aims to improve the model's ability to correctly identify delayed flights mainly for the less common
# delay categories, at the potential cost of increased false positives.
##################################################################################################################
from sklearn.metrics import fbeta_score

def custom_classification_report(y_true, y_pred, beta=2):
    classes = np.unique(y_true)
    report = f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F{beta}-score':<12}{'Support':<12}\n"
    report += "-" * 58 + "\n"

    for cls in classes:
        y_true_cls = (y_true == cls)
        y_pred_cls = (y_pred == cls)

        tp = np.sum((y_true_cls) & (y_pred_cls))
        fp = np.sum((~y_true_cls) & (y_pred_cls))
        fn = np.sum((y_true_cls) & (~y_pred_cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f_beta = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true_cls)

        report += f"{cls:<10}{precision:<12.2f}{recall:<12.2f}{f_beta:<12.2f}{support:<12}\n"

    return report + "\n"


def evaluate_model_with_recall_focus_class2(estimator, Xt_test, yt_test, X_val, y_val, beta=2, n_classes=3):
    """
    Modifications to prioritize recall, especially for class 2.

    Changes:
        1. Adjust thresholds ONLY for class 2.
        2. Use F2 score instead of F1 score.

    Notes:
        1. Adjusting thresholds for class 2 allows fine-tuning of the decision boundary
           to potentially improve recall for this minority class.
        2. F2 score weighs recall higher than precision, aligning with the goal of
           minimizing false negatives.
    """
    # Evaluate on validation set
    val_pred = estimator.predict(X_val)
    val_f2 = fbeta_score(y_val, val_pred, beta=beta, average='weighted')
    val_accuracy = accuracy_score(y_val, val_pred)

    print(f"Validation F{beta} Score: {val_f2:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Predict probabilities for multiclass classification
    y_pred_proba = estimator.predict_proba(Xt_test)

    # Calculate original predictions
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Adjust the threshold for class 2 based on F2 score optimization
    precision, recall, thresholds = precision_recall_curve(yt_test == 2, y_pred_proba[:, 2])
    f2_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    best_threshold = thresholds[np.argmax(f2_scores)]

    # Adjust predictions for class 2 using the best threshold
    y_pred_adjusted = np.copy(y_pred)
    y_pred_adjusted[y_pred_proba[:, 2] >= best_threshold] = 2

    # Calculate F2 score and accuracy for adjusted predictions
    test_f2_macro = fbeta_score(yt_test, y_pred_adjusted, beta=beta, average='macro')
    test_f2_weighted = fbeta_score(yt_test, y_pred_adjusted, beta=beta, average='weighted')
    test_accuracy = accuracy_score(yt_test, y_pred_adjusted)

    # Calculate and display F2 scores for each class
    f2_per_class = fbeta_score(yt_test, y_pred_adjusted, beta=beta, average=None)

    # Display F2 scores for each class
    class_labels = np.unique(yt_test)  # Unique class labels
    print("\nF2 Scores per Class:")
    for cls, f2_score in zip(class_labels, f2_per_class):
        print(f"Class {cls}: F2 Score = {f2_score:.4f}")

    # Display macro and weighted F2 scores
    print(f"\nMacro-Averaged F2 Score: {test_f2_macro:.4f}")
    print(f"Weighted-Averaged F2 Score: {test_f2_weighted:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")

    # Classification report
    print("\nCustom Classification Report with F2 scores:")
    print(custom_classification_report(yt_test, y_pred_adjusted, beta=beta))

    # Compute PR AUC and ROC AUC
    # Plot confusion matrix and curves

    # Compute PR AUC and ROC AUC using one-vs-rest for each class
    pr_auc_values = []
    roc_auc_values = []

    for cls in np.unique(yt_test):
        # Convert to binary: 1 if the class is `cls`, 0 otherwise
        y_test_binary = (yt_test == cls).astype(int)

        # Get class-specific predicted scores (probabilities for PR AUC and ROC AUC)
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
    conf_matrix = confusion_matrix(yt_test, y_pred_adjusted)
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

    # Return evaluation metrics
    results = {
        'val_f2_score': val_f2,
        'val_accuracy_score': val_accuracy,
        'f2_per_class': f2_per_class,
        'f2_score_macro': test_f2_macro,
        'f2_score_weighted': test_f2_weighted,
        'accuracy_score': test_accuracy,
        'pr_auc_macro': pr_auc_macro,
        'roc_auc_macro': roc_auc_macro,
        'pr_auc_weighted': pr_auc_weighted,
        'roc_auc_weighted': roc_auc_weighted,
        'confusion_matrix': conf_matrix,
        'best_threshold': best_threshold
    }

    return results


# In[ ]:


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


# In[ ]:


###################################################################################################
# Generic Function to plot built-in feature importances from a trained classifier
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


###################################################################################################
# Generic Function to plot built-in feature importances from a trained classifier v.s. feature
# importances from permutation importance
###################################################################################################
from sklearn.inspection import permutation_importance

def plot_feature_importances_w_permutation_importance_vs_built_in(pipeline, X_test, y_test):
    """
    Computes and plots permutation feature importance and a classifier's built-in feature importances.

    Parameters:
        pipeline : sklearn.pipeline.Pipeline
            The trained pipeline containing the preprocessor, feature selector, and classifier.
        X_test : pandas.DataFrame or numpy.ndarray
            The test dataset used for evaluating feature importances.
        y_test : pandas.Series or numpy.ndarray
            The true labels for the test dataset.
    """
    # Extract the trained Classifier model from the pipeline
    model = pipeline.named_steps['classifier']

    # Extract the preprocessor from the pipeline
    preprocessor = pipeline.named_steps['preprocessor']

    # Get feature names after preprocessing
    feature_names = []
    for name, trans, columns in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(trans.get_feature_names_out(columns).tolist())
        elif name == 'cat':
            feature_names.extend(columns)

    # Transform X_test with the preprocessor
    X_test_transformed = preprocessor.transform(X_test)

    # Check if the pipeline has a feature selection step
    if 'feature_selection' in pipeline.named_steps:
        # Get selected feature indices from the feature selector (RFE)
        selected_features = pipeline.named_steps['feature_selection'].get_support()

        # Filter feature names based on selected features
        selected_feature_names = [name for name, selected in zip(feature_names, selected_features) if selected]

        # Compute permutation importance for the selected features
        X_test_transformed_selected = X_test_transformed[:, selected_features]
        results = permutation_importance(model, X_test_transformed_selected, y_test, scoring='f1_weighted')
    else:
        # If there is no feature selection, all features are used
        selected_feature_names = feature_names

        # Compute permutation importance for all features
        results = permutation_importance(model, X_test_transformed, y_test, scoring='f1_weighted')

    #################################################################################################
    # Permutation Feature Importance
    #################################################################################################
    # Print permutation feature importances
    print("\nPermutation Feature Importances:")
    for idx, name in enumerate(selected_feature_names):
        print(f"{name}: {results.importances_mean[idx]:.4f} +/- {results.importances_std[idx]:.4f}")

    # Plot permutation feature importances
    plt.figure(figsize=(15, 6))
    sorted_idx = results.importances_mean.argsort()
    plt.barh(range(len(results.importances_mean)), results.importances_mean[sorted_idx])
    plt.yticks(range(len(results.importances_mean)), [selected_feature_names[i] for i in sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Permutation Importances (Classifier)")
    plt.tight_layout()
    plt.show()

    #################################################################################################
    # Classifier's Built-in Feature Importances
    #################################################################################################
    # Get built-in feature importances from the Classifier
    c_importances = model.feature_importances_

    # Print Classifier's built-in feature importances
    print("\nClassifier's Built-in Feature Importances:")
    for idx, name in enumerate(selected_feature_names):
        print(f"{name}: {c_importances[idx]:.4f}")

    # Plot Classifier's built-in feature importances
    plt.figure(figsize=(15, 6))
    sorted_idx = c_importances.argsort()
    plt.barh(range(len(c_importances)), c_importances[sorted_idx])
    plt.yticks(range(len(c_importances)), [selected_feature_names[i] for i in sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Classifier's Built-in Feature Importances")
    plt.tight_layout()
    plt.show()


# In[ ]:


###################################################################################################
# Generate class weight dictionaries with multipliers applied to class 2
###################################################################################################
from sklearn.utils.class_weight import compute_class_weight

def generate_class_weights_for_class2(y, multipliers):
    """
    Generates class weight dictionaries with specific multipliers applied to class 2.

    Parameters:
        y : series
            Target labels from which to compute class weights.
        multipliers : list of floats
            A list of multipliers to apply to the weight of class 2.

    Returns:
        List[dict]
            A list of dictionaries, where each dictionary contains the class weights.
            For each dictionary, the weight for class 2 is adjusted by the corresponding multiplier.
    """

    # Compute balanced class weights for each unique class in y
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

    # Create a dictionary mapping each class to its corresponding computed weight
    weight_dict = dict(zip(np.unique(y), weights))

    # Balanced class weights
    print("Balanced Class Weights:")
    for class_label, weight in weight_dict.items():
        print(f"Class {class_label}: Weight {weight}")

    # Step 3: Generate a list of weight dictionaries
    return [
        {k: (v if k != 2 else v * mult) for k, v in weight_dict.items()}  # Apply multiplier to class 2
        for mult in multipliers  # Iterate through each multiplier
    ]


# In[ ]:


###################################################################################################
# Generate class weight dictionaries with multipliers applied to all classes
###################################################################################################
from sklearn.utils.class_weight import compute_class_weight
from itertools import product

def generate_class_weights_for_all(y, multipliers):
    """
    Generates class weight dictionaries with specific multipliers applied to each class.

    Parameters:
        y : array-like
            Target labels from which to compute class weights.
        multipliers : dict
            A dictionary where keys are class labels and values are lists of multipliers
            to apply to the weights of the respective classes.

    Returns:
        List[dict]
            A list of dictionaries, where each dictionary contains the class weights.
            For each dictionary, the weights for each class are adjusted by the corresponding multipliers.
    """

    # Compute balanced class weights for each unique class in y
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)

    # Create a dictionary mapping each class to its corresponding computed weight
    weight_dict = dict(zip(classes, weights))

    # Print balanced class weights
    print("Balanced Class Weights:")
    for class_label, weight in weight_dict.items():
        print(f"Class {class_label}: Weight {weight}")

    # Generate all combinations of multipliers
    multiplier_combinations = list(product(*[multipliers.get(cls, [1.0]) for cls in classes]))

    # Generate a list of weight dictionaries
    result = []
    for multiplier_combo in multiplier_combinations:
        new_weights = weight_dict.copy()
        for cls, mult in zip(classes, multiplier_combo):
            new_weights[cls] *= mult
        result.append(new_weights)

    return result


# In[ ]:


###################################################################################################
# Generic function that calculates the F1 score for class 2, adjusts the prediction
# threshold, and returns the adjusted predictions
###################################################################################################

from sklearn.metrics import f1_score, precision_recall_curve

def adjust_class_2_threshold(classifier, X_test, y_test):
    """
    Adjusts the classification threshold for class 2 based on F1 score optimization.

    Parameters:
        - classifier: Trained classifier model.
        - X_test: Feature data for testing.
        - y_test: True labels for testing.

    Returns:
        - class_2_f1_adjusted: Adjusted F1 score for class 2 based on adjusted predictions with optimized threshold.
        - class_2_f1: Original F1 score for class 2.
    """
    # Predict with the stacking classifier
    y_pred = classifier.predict(X_test)

    # Calculate the original F1 score for class 2
    class_2_f1 = f1_score(y_test, y_pred, average=None)[2]

    # Get predicted probabilities
    y_pred_proba = classifier.predict_proba(X_test)

    # Calculate precision, recall, and thresholds for class 2
    precision, recall, thresholds = precision_recall_curve(y_test == 2, y_pred_proba[:, 2])

    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Find the best threshold that maximizes the F1 score
    best_threshold = thresholds[np.argmax(f1_scores)]

    # Adjust predictions based on the best threshold
    y_pred_adjusted = np.copy(y_pred)
    y_pred_adjusted[y_pred_proba[:, 2] >= best_threshold] = 2

    # Calculate adjusted F1 score based on adjusted predictions for best threshold
    class_2_f1_adjusted = f1_score(y_test, y_pred_adjusted, average=None)[2]

    print("\nAfter threshold adjustment for class 2, classification report:")
    print(classification_report(y_test, y_pred_adjusted))

    return class_2_f1_adjusted, class_2_f1


# In[8]:


###################################################################################################
# Generic functions to persist and load models with joblib
###################################################################################################
get_ipython().system('pip install joblib')
from joblib import dump, load


def save_model(model, filename):
    """
    Save a model to a .pkl file using joblib.

    Parameters:
        model: Trained model to be saved.
        filename: Name of the file to save the model to (should end with .pkl).
    """
    dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    Load a model from a .pkl file using joblib.

    Parameters:
        filename: Name of the file to load the model from (should end with .pkl).

    Returns:
        Loaded model.
    """
    model = load(filename)
    print(f"Model loaded from {filename}")
    return model

# Example usage:
# save_model(your_model, 'model.pkl')
# loaded_model = load_model('model.pkl')


# In[ ]:




