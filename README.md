# 💼📉 Bank-Customer-Churn-Prediction-and-ML-Model-Comparison

A complete customer churn analytics & prediction project built using **Python, Scikit-Learn, and XGBoost**, applying advanced ML techniques to forecast churn in the banking sector. Includes rigorous testing of multiple models, imbalance handling, real metrics evaluation (ROC-AUC / PR-AUC), and a future-ready Streamlit interface for real-time prediction.

---

## 🧩 **1. Project Workflow Overview**

This project follows an industry-standard ML pipeline:

- **Python (Pandas / Numpy)**: Data cleaning & preprocessing  
- **EDA**: Customer behaviour analysis  
- **Imbalance Handling**: SMOTE, ADASYN, Undersampling, Class Weights  
- **Model Training**: 6 different ML algorithms tested  
- **Model Evaluation**: ROC-AUC, PR-AUC, Recall-oriented assessment  
- **Deployment Prep**: Streamlit app for interactive churn predictions  

---

## 🧹 **2. Data Cleaning & Preparation (Python)**

### 🔧 **Key Cleaning Operations**
- Removed duplicates  
- Filled missing values  
- Label encoding & one-hot encoding  
- Numerical scaling using StandardScaler  
- Final processed dataset exported for reproducibility  

### 📊 **Model Features**
| Feature | Description |
|---------|-------------|
| CreditScore | Customer credit rating |
| Age | Customer age |
| Tenure | Years with bank |
| Balance | Current account balance |
| NumOfProducts | Number of bank services used |
| HasCrCard | Credit card status |
| IsActiveMember | Engagement indicator |
| Gender | Male/Female |
| Geography | Country |
| EstimatedSalary | Salary |

---

## ⚖️ **3. Handling Class Imbalance**

Dataset distribution:
- **80%** Non-Churn  
- **20%** Churn  

Applied strategies:
- Random Undersampling  
- SMOTE  
- ADASYN  
- Class weights  
- XGBoost scale_pos_weight  

This ensured prediction fairness toward minority class (churners).

---

## 🧠 **4. Machine Learning Models Used**

Implemented and compared:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- **XGBoost (winner)**  

Each with:
- Base model  
- SMOTE version  
- ADASYN version  
- Undersampling  
- Class Weighting  

Hyperparameter tuning applied using:
- GridSearchCV(cv=5, scoring=[precision, recall, f1], refit='recall')
- This ensures the models are optimized to **maximize recall for churn detection**, since retaining customers is more important than false positives.

---

## 📊 **5. Model Performance**

### 🏆 Final Best Model: **XGBoost**

| Metric | Score |
|--------|-------|
| Recall (Churn class) | **76%** |
| ROC-AUC | **88%** |
| PR-AUC | **71%** |

➡ Selected as final model due to best performance in identifying churners while maintaining strong overall ranking ability.

---

## 📁 **6. Dataset Description**

Includes customer details and churn indicator:

- Age  
- CreditScore  
- Tenure  
- Balance  
- Estimated Salary  
- Geography  
- Gender  
- Number of products  
- Active membership  
- Has/Doesn’t have credit card  
- Target column: `Exited` (1 = churn, 0 = stay)

---

## 💡 **7. Key Insights From Analysis**

✔ Customers with low activity & engagement churn more  
✔ Higher number of products → reduced churn  
✔ Salary alone is not a strong predictor  
✔ Certain geographical regions have more churn tendency  
✔ Middle-aged customers show higher churn patterns  

---

## 🚀 **8. Future Enhancements**

- Add SHAP or LIME for interpretability  
- Create Streamlit UI for model demo  
- Deploy model via API endpoint  
- Continuously train model on fresh data  
- Add temporal behaviour features (e.g., monthly transactions)  

---

> 🙌 **Built with strong ML methodology, fair sampling strategies, and statistical model evaluation to support business-driven churn reduction strategies.**

---

### 🔗 **Connect**

For collaboration or suggestions — feel free to reach out!



