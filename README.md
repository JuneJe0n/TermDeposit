
# Term Deposit Pattern Recognition
💻 [Full Codes](./code.ipynb) <br>
📄 [Full Report](./report.pdf)
      
## Overview
This project aims to predict whether a client will subscribe to a bank term deposit using machine learning techniques.
- Task: Binary classification (y ∈ {0,1})
- Goal: Maximize F1-score
- Evaluation:
  - Performance is measured on a hidden test set 
  - Only part of the test labels are accessible during development

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
**The dataset is highly imbalanced, with a ratio of approximately 1 : 8 between positive and negative classes.**

### Problem of Imabalanced Dataset
In imbalanced datsets:
- One class dominates → model predicts majority class
- This causes:
      - Low recall for minority class
      - Unbalanced precision/recall trade-off

As a result:
- **Even if accuracy is high, F1-score remains low because it requires both precision and recall to be balanced.
**
