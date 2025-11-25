# 💼📉 Bank-Customer-Churn-Prediction-and-ML-Model-Comparison

A complete churn prediction project developed using **Python and Scikit-Learn**, involving data preprocessing, feature encoding, scaling, and multiple imbalance handling techniques such as **SMOTE, ADASYN, undersampling, and class weighting**. The project applies various machine learning algorithms, including logistic models, tree-based methods, and boosting techniques such as **AdaBoost, Gradient Boosting, and XGBoost**, while carefully addressing and preventing data leakage using proper train–test split pipelines. Model performance is evaluated using metrics like ROC-AUC, PR-AUC, Recall, and F1-Score to enable a structured comparison and determine the most effective approach for customer churn prediction.

---

## 🧩 **1. Project Workflow Overview**

This project follows an industry-standard ML pipeline:

- **Python (Pandas / Numpy)** → Data cleaning & preprocessing  
- **Data Leakage Prevention** → Ensuring transformations & resampling are applied only on training data using pipelines  
- **Imbalance Handling** → SMOTE, ADASYN, Undersampling, Class Weights  
- **Model Training** → Trained 6 different ML algorithms + sampling variants  
- **Model Evaluation** → Recall, F1, ROC-AUC, PR-AUC metrics  
- **Final Model Selection** → XGBoost with highest churn detection capability  

---

## 🧹 **2. Data Cleaning & Preparation (Python)**

### 🔧 **Key Cleaning Operations**
- Removed duplicates  
- Handled missing values  
- Label encoding & one-hot encoding for categorical columns  
- Numerical feature scaling using StandardScaler  
- Saved intermediate cleaned datasets for reproducibility  

### 📊 **Final Features Used**
| Feature | Description |
|---------|-------------|
| CreditScore | Customer credit rating |
| Age | Customer age |
| Tenure | Years with bank |
| Balance | Current account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Credit card status |
| IsActiveMember | Customer engagement |
| Gender | Male/Female |
| Geography | Country |
| EstimatedSalary | Income level |

---

## ⚖️ **3. Handling Class Imbalance**

Dataset distribution:  
- **80%** — Not churn (class 0)  
- **20%** — Churn (class 1)  

To avoid bias toward the majority class, applied:

- Random Undersampling  
- SMOTE  
- ADASYN  
- Class Weights  
- XGBoost scale_pos_weight  

Objective: improve recall of churners.

---

## 🧠 **4. Machine Learning Models Used**

Implemented and compared:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- **XGBoost (best)**  

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



