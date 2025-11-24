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


---

## 📊 **5. Model Performance**

### 🏆 Final Best Model: **XGBoost**

| Metric | Score |
|--------|-------|
| Recall (Churn class) | **76%** |
| ROC-AUC | **88%** |
| PR-AUC | **71%** |

➡ Selected as final model due to best discrimination & real business recall performance.

---

## 📁 **6. Dataset Description**

Includes customer attributes & churn status:

- Age  
- CreditScore  
- Tenure  
- Balance  
- Salary  
- Geography  
- Product usage  
- Customer engagement  
- `Exited` flag → churn (1) or not (0)

---

## 💡 **7. Key Insights**

✔ Older customers tend to churn more  
✔ Low activity → higher churn risk  
✔ Customers with only 1 product have higher churn rate  
✔ Salary itself not a strong predictor  
✔ Geography influences churn behaviour  

---

## 🚀 **8. Future Enhancements**

- Add SHAP explainability  
- Deploy on Streamlit Cloud  
- Integrate internal bank database  
- Real-time refresh on new customer events  
- API endpoint for churn monitoring pipeline  

---

> 🙌 **Built with real applied ML, model evaluation depth, and practical churn interpretation to support bank business decisions.**

---

### 🔗 **Connect**

Feel free to reach out for collaboration or improvements to extend churn prediction capabilities.


