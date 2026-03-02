# Credit Card Fraud Detection — Neural Network Pipeline

Neural-network approach for detecting fraudulent credit card transactions using an imbalanced real-world dataset. The project presents a complete machine-learning workflow from raw data to predictions.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Overview

The goal is to estimate the probability that a transaction is fraudulent and to make decisions based on that risk. Because fraud is rare, the process focuses on handling severe class imbalance and evaluating performance in a way that reflects real operational needs.

The pipeline follows a structured sequence:

1. Load and inspect the data  
2. Prepare features and target  
3. Normalize key numeric variables  
4. Split data into training and test sets  
5. Address class imbalance  
6. Train a neural network classifier  
7. Monitor performance during training  
8. Evaluate on unseen data  
9. Adjust the decision threshold  
10. Identify highest-risk transactions  

---

## Dataset Characteristics

The dataset contains anonymized credit card transactions with:

- A binary label indicating fraud or legitimate activity  
- Numerical features derived from transaction information  
- Strong class imbalance (fraud cases are extremely rare)  
- Wide variation in transaction amounts  

Because of anonymization, domain interpretation of individual features is limited; the model relies on statistical patterns.

---

## Pipeline Description

### 1. Data Loading and Initial Inspection

The dataset is loaded and basic checks are performed to confirm it is usable:

- Verify size and structure  
- Examine the distribution of the fraud label  
- Check for missing values  
- Inspect how transaction amounts are distributed  

These steps ensure the data is intact and reveal critical properties such as imbalance and skewness.

---

### 2. Feature and Target Preparation

The fraud indicator is separated from the explanatory variables.

- The target represents whether a transaction is fraudulent  
- All other columns are treated as predictors  

This creates the inputs required for supervised learning.

---

### 3. Normalization of Key Variables

Certain numeric variables (time and transaction amount) can have large ranges that destabilize training. These variables are standardized so that they have comparable scales.

Normalization improves optimization efficiency and prevents large-magnitude features from dominating the learning process.

---

### 4. Train/Test Split with Preserved Class Ratio

The dataset is divided into two parts:

- A training set used to learn model parameters  
- A test set reserved for final evaluation  

The split preserves the original fraud proportion in both subsets, ensuring that performance estimates reflect real conditions.

---

### 5. Handling Severe Class Imbalance

Because fraudulent transactions are vastly outnumbered, the training process is adjusted so that errors on fraud cases carry greater importance than errors on legitimate transactions.

This encourages the model to learn patterns associated with fraud rather than defaulting to majority predictions.

---

### 6. Neural Network Model

A feedforward neural network is used for classification.

Key characteristics:

- Multiple layers allow the model to learn nonlinear relationships  
- Regularization techniques reduce overfitting  
- The output represents a probability of fraud  

This architecture is suitable for structured numerical data.

---

### 7. Training with Validation Monitoring

The model is trained iteratively on the training data while periodically evaluating performance on a held-out validation subset.

Training stops automatically when improvement on validation performance stalls. This prevents overfitting and selects the model that generalizes best rather than the one that simply fits the training data most closely.

Performance measures emphasize detection quality for rare events rather than overall accuracy.

---

### 8. Evaluation on Unseen Test Data

After training, the model is applied to the test set, which has not influenced any training decisions.

The model outputs fraud probabilities for each transaction. These predictions are used to compute performance measures that reflect how well the model separates fraudulent from legitimate activity.

Using a never-seen dataset provides an unbiased estimate of real-world performance.

---

### 9. Decision Threshold Adjustment

A probability must be converted into a binary decision (fraud vs legitimate). Instead of using a fixed cutoff, the threshold is tuned to achieve a desired trade-off:

- Higher recall → more fraud detected but more false alarms  
- Higher precision → fewer false alarms but more missed fraud  

This step tailors the model to operational priorities.

---

### 10. Identification of Highest-Risk Transactions

Transactions are ranked by predicted fraud probability.

Examining the highest-risk cases provides:

- A practical view of what the model flags  
- A sanity check on whether true fraud appears among top predictions  
- Insight into how the system might be used in real monitoring scenarios  

---

## Why This Approach Works for Fraud Detection

Fraud detection differs from typical classification tasks because:

- Fraud events are extremely rare  
- Missing fraud is costly  
- False alarms also carry operational and customer costs  

This pipeline addresses those challenges by:

- Compensating for imbalance during training  
- Using probability outputs rather than hard decisions  
- Evaluating with metrics appropriate for rare events  
- Allowing threshold tuning based on operational needs  

