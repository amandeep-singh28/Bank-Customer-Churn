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

### 🧠 Model Selection Rationale & Challenges

This project focuses on a binary classification problem where the objective is to predict whether a customer will stay with the bank or churn (leave). I began by experimenting with three base models — **Logistic Regression**, **Decision Tree**, and **Random Forest**. Below is a detailed explanation of the challenges encountered with each model and the workflow followed to understand and improve their performance.

---

### (i) Logistic Regression

Initially, I began by training a Logistic Regression model and evaluating its performance using the classification report. The model showed excellent metrics (accuracy, precision, recall, F1-score) for class 0 (non-churn), but performed very poorly for class 1 (churn). This clearly indicated that the model was biased toward predicting the majority class.

#### 📊 Classification Report (Logistic Regression)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.83 | 0.61 |
| Recall | 0.96 | 0.21 |
| F1-score | 0.89 | 0.32 |
| Support | 2389 | 611 |

It is clear from the above table that Class 0 significantly outperforms Class 1 across all metrics. This indicates that the model is biased toward the majority class and struggles to correctly identify churned customers due to the dataset’s imbalance.

---

To verify this, I examined the distribution of the target variable `Exited`, which showed a clear imbalance:
`y.value_counts()`
| Exited (y) | Count |
|------------|-------|
| 0 (Not churn) | 7960 |
| 1 (Churn) | 2037 |

---

### (ii) Logistic Regression + Undersampling Approach

To address the class imbalance, I applied **Random Undersampling**, which reduces the majority class by randomly removing samples to balance both classes. However, this approach may also discard useful data from the majority class, potentially affecting overall performance.<br>
`from imblearn.under_sampling import RandomUnderSampler`  
`steps = [("preprocess", preprocessor),`  
`         ("undersampling", RandomUnderSampler(random_state=42)),`  
`         ("logistic_regression", LogisticRegression(random_state=42))]`


#### 📊 Classification Report (Logistic Regression + Undersampling)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.91 | 0.40 |
| Recall | 0.73 | 0.72 |
| F1-score | 0.81 | 0.52 |
| Support | 2389 | 611 |

It can be observed that the recall value for Class 1 increased significantly (from 0.21 to 0.72), meaning the model is now better at catching churners. However, precision for Class 1 decreased, indicating more false churn predictions. This tradeoff is expected with undersampling since valuable data from Class 0 is removed.

---

### (iii) Logistic Regression + SMOTE

SMOTE (Synthetic Minority Oversampling Technique) is an oversampling method that generates new synthetic samples for the minority class rather than simply duplicating existing records. It does this by interpolating between neighboring minority class samples using the K-Nearest Neighbours approach. This allows the model to learn more generalized minority patterns and typically performs better than both random oversampling and undersampling.<br>
`from imblearn.over_sampling import SMOTE`  
`steps = [("preprocess", preprocessor),`  
`         ("smote", SMOTE(random_state=42)),`  
`         ("logistic_regression", LogisticRegression(random_state=42))]`

#### 📊 Classification Report (Logistic Regression + SMOTE)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.91 | 0.40 |
| Recall | 0.72 | 0.71 |
| F1-score | 0.81 | 0.51 |
| Support | 2389 | 611 |

Compared to undersampling, SMOTE retains all original data and generates artificial churn examples, leading to a much more balanced learning process. The recall for Class 1 remains significantly higher compared to the baseline model, indicating strong improvement in detecting churners.

---

### (iv) Logistic Regression + ADASYN

ADASYN (Adaptive Synthetic Sampling) is similar to SMOTE, but instead of generating an equal number of synthetic samples around each minority instance, it focuses on generating more samples for minority points that are harder to learn — specifically those located in regions dominated by the majority class. This makes the model pay more attention to difficult minority examples.<br>
`from imblearn.over_sampling import ADASYN`  
`steps = [("preprocess", preprocessor),`  
`         ("adasyn" ADASYN(random_state=42)),`  
`         ("logistic_regression", LogisticRegression(random_state=42))]`


#### 📊 Classification Report (Logistic Regression + ADASYN)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.91 | 0.38 |
| Recall | 0.68 | 0.75 |
| F1-score | 0.78 | 0.50 |
| Support | 2389 | 611 |

ADASYN further improves recall for Class 1 (0.75), but precision decreases slightly. The model becomes more aggressive in predicting churn, catching more true churners, but also producing more false alarms. This behavior is expected due to the increased focus on hard-to-classify minority instances.






---

## ✈️ **2. Detailed Project Explanation**

This section describes the end-to-end workflow followed in this project, from raw data to final model comparison.

### 🥇 (i). Importing and Understanding the Dataset
- Loaded the bank customer dataset containing features such as:
  - Customer demographics (Age, Gender, Geography)
  - Account information (Balance, Tenure, Number of Products, HasCrCard, IsActiveMember)
  - Financial details (CreditScore, EstimatedSalary)
  - Target variable: **Exited** (1 = churn, 0 = not churn)
- Performed basic checks:
  - Shape of the dataset  
  - Data types of each column  
  - Presence of null values and duplicates

---

### 🧼 (ii). Data Cleaning

- Handled missing values:
  - Replaced null values in the `Surname` column using the mode (most frequent value).
  - Replaced missing values in the `Age` column using the mean of the age distribution.

- Standardized inconsistent categorical values:
  - Converted `'FRA'` and `'French'` entries in the `Geography` column to `'France'` for consistency.

- Cleaned numeric fields:
  - Removed the `'€'` symbol from the `EstimatedSalary` column to ensure numerical formatting.

---

### 📉 3. Outlier Detection & Handling

- Visualized all numerical features using boxplots to detect potential outliers.  
- Used the IQR (Interquartile Range) method along with list comprehension to isolate and display outlier values for review.  
- Identified a few invalid entries in `EstimatedSalary` (negative values) and removed them to maintain data integrity.  
- Plotted histograms for each numerical feature to understand their distributions and validate the impact of outlier removal.

---

### 🔁 4. Feature Encoding & Preprocessing

- Applied One-Hot Encoding on the `Geography` column using `pd.get_dummies`, while dropping the first dummy column to avoid the dummy variable trap and reduce multicollinearity.  
- Converted binary categorical columns (`Gender`, `HasCrCard`, `IsActiveMember`) into numeric form using `LabelEncoder` for efficient model ingestion.














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



