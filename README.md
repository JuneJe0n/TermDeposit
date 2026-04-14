
# Term Deposit Pattern Recognition
    
## Overview
This project aims to predict whether a client will subscribe to a bank term deposit using machine learning techniques.
- Task: Binary classification (y ∈ {0,1})
- Goal: Maximize F1-score
- Evaluation:
  - Performance is measured on a hidden test set 
  - Only part of the test labels are accessible during development
 
💻 [Full Codes](./code.ipynb) <br>
📄 [Full Report](./report.pdf)

## Project Structure
```
├── EDA
├── Data Preprocessing
├── Feature Selection
├── Model Training
├── Evaluation & Analysis
```

## 1. Data Exploration (EDA)
Through EDA, we identified a critical issue: <br>
```
The dataset is highly imbalanced,
with a ratio of approximately 1 : 8 between positive and negative classes.
```

### ► Problem of Imabalanced Dataset
In imbalanced datsets:
- One class dominates → model predicts majority class
- This causes:
      - Low recall for minority class
      - Unbalanced precision/recall trade-off

As a result: <br>

**Even if accuracy is high, F1-score remains low because it requires both precision and recall to be balanced.**

### ► Modeling Strategy
This insight directly shaped our approach:
- ❌ Do not rely on accuracy
- ✅ Focus on F1-score optimization
- ✅ Design strategies specifically for class imbalance

Based on this, we explored two approaches:<br>
```
1. Resampling-based methods (SMOTE + ensemble) 
2. Class-weighted learning (LightGBM)
```
The entire modeling pipeline was built around solving this issue.

## 2. Data Preprocessing
We designed a comprehensive preprocessing pipeline tailored to the dataset, including missing value and outlier handling, feature encoding, normalization, and feature selection.

To keep this README concise, we highlight only the key design choices, while the full implementation can be found in the [full code](./code.ipynb).

### ► Handle Missing Values
We observed missing values in multiple categorical features, including ```job, marital, education, default, housing, loan, cons.conf.idx```<br>
Rather than applying a single imputation strategy, we designed **feature-specific methods based on data distribution.**






