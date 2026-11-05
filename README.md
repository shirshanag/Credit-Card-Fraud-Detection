# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Logistic Regression, Random Forest and Support Vector Machine (SVM) classifiers. Class imbalance is handled with undersampling, and the models are compared on accuracy, precision, recall and F1-score.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Overview

Fraudulent transactions make up a tiny fraction of all credit card activity, so a model that predicts "legitimate" every time can still look accurate while catching no fraud. This project addresses that problem by:

- Balancing the data with undersampling
- Training three different classifiers
- Evaluating them on both train and test data to check for overfitting
- Comparing them on metrics that matter for fraud: precision and recall

## Dataset

- Source: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]
- Features: 30 numerical features per transaction
- Target: binary label (`0` = legitimate, `1` = fraud)
- The original data is heavily imbalanced, so the legitimate class was undersampled to match the fraud class.

## Approach

1. **Data loading and exploration**: inspect shape, class distribution and missing values.
2. **Undersampling**: randomly sample legitimate transactions to balance the two classes.
3. **Preprocessing**: feature scaling and a train-test split.
4. **Model training**: Logistic Regression, Random Forest and SVM.
5. **Evaluation**: accuracy, precision, recall and F1-score on both train and test sets.
6. **Comparison**: compare train and test scores to detect overfitting.

## Results

### Test Data Performance

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9347   | 0.9778    | 0.8889 | 0.9312   |
| Random Forest       | 0.9397   | 1.0000    | 0.8788 | 0.9355   |
| SVM                 | 0.9347   | 1.0000    | 0.8687 | 0.9297   |

### Train Data Performance

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9433   | 0.9704    | 0.9137 | 0.9412   |
| Random Forest       | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| SVM                 | 0.9547   | 0.9972    | 0.9112 | 0.9523   |

## Key Findings

- **Random Forest** had the highest test accuracy (93.97%) and F1-score (0.9355), with 100% precision, so every transaction it flagged as fraud was actually fraud.
- **Logistic Regression** had the best recall on test data (88.89%), so it caught the most fraud cases, and its train and test scores were the closest.
- **SVM** also reached 100% precision on test data but had the lowest recall (86.87%).
- **Random Forest overfit the training data** (100% on every train metric), so test results were used as the real benchmark.
- In fraud detection, precision and recall trade off against each other. Choose Random Forest or SVM to avoid false alarms, or Logistic Regression to catch more fraud.

## Tech Stack

- Python 3
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn
- Jupyter Notebook

## Installation

```bash
git clone https://github.com/shirshanag/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
```

## Usage

1. Download the dataset and place it in the project folder (or the `data/` directory).
2. Open the notebook:

```bash
jupyter notebook
```

3. Run all cells to reproduce the preprocessing, training and evaluation.

## Project Structure

```
.
├── data/                 # Dataset (not included)
├── notebook.ipynb        # Main analysis and model training
├── requirements.txt      # Python dependencies
└── README.md
```

## Future Improvements

- Try oversampling (SMOTE) and compare it with undersampling
- Tune hyperparameters with GridSearchCV or RandomizedSearchCV
- Add gradient boosting models such as XGBoost or LightGBM
- Add ROC-AUC and Precision-Recall curves
- Use cross-validation for more reliable estimates
- Deploy the best model as a simple web app or API

## Author

**Shirsha Nag**
[LinkedIn](https://www.linkedin.com/in/shirsha-nag-b4aa87219/) | [GitHub](https://github.com/shirshanag)
