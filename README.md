# Kaggle ML Project – Titanic Survival Prediction

## 1. Problem Statement
Predict passenger survival state using given attributes.

- Task type: Classification (Survived 0/1)
- Evaluation metric (Kaggle): Prediction Accuracy

---

## 2. Dataset
- Source: Kaggle Titanic
- Input: Passenger attributes from train.csv (e.g., `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`)
- Target: Survived
- Dataset split strategy:
  - Training / validation split based on stratified K-fold

---

## 3. Exploratory Data Analysis (EDA)
Key observations from EDA:
- Data distribution and imbalance
- Missing values / outliers
- Feature-target relationships

Relevant plots can be found in:
notebooks/00_eda.ipynb

---

## 4. Approach

### 4.1 Baseline
A simple baseline model was implemented to establish a performance reference.

- Model: Logistic Regression
- Purpose: Sanity check and lower-bound performance

### 4.2 Improved Model
The improved model was built using PyTorch with the following enhancements. A decision tree based model (XGBoost) was also included for comparison:

- Feature engineering / preprocessing
- Hyperparameter optimization
- Comparison with XGBoost model

---

## 5. Evaluation & Error Analysis
Evaluation is performed using scikit-learn metrics after PyTorch inference.

- Metrics used:
  - accuracy / AUC score
- Error analysis:
  - Class-wise performance
  - Common failure cases
  - Possible data or model limitations

---

## 6. Results

| Model | Metric | Kaggle Score | Notes |
|-------|--------|--------------|-------|
| All female guess | accuracy | 0.76555 | gender_submission.csv |
| Baseline | accuracy | 0.76315 | Logistic Classifier |
| Improved I | accuracy | 0.77033 | Baseline + FE |
| Final | accuracy | 0.78947 | XGBoost + FE |

Leaderboard snapshot date: <2026-01-11>

---

## 7. How to Run

### 7.1 Environment setup
pip install -r requirements.txt

or

pip install -e .

### 7.2 Training
python src/train.py

### 7.3 Generate submission
python src/predict.py

The generated file will be saved under:
outputs/submissions/

---

## 8. Project Structure
```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── outputs/
└── README.md
```
---

## 9. Limitations & Future Work
- Model generalization limitations
- Data-related constraints
- Potential improvements for future iterations

---

## 10. Notes
This project is designed as an interview-ready ML portfolio piece, emphasizing:
- Clear problem formulation
- Reproducible experiments
- Explicit baseline comparison
- Interpretable evaluation
