# 💼📉 Bank-Customer-Churn-Prediction-and-ML-Model-Comparison

A complete churn prediction project developed using **Python and Scikit-Learn**, involving data preprocessing, feature encoding, scaling, and multiple imbalance handling techniques such as **SMOTE, ADASYN, undersampling, and class weighting**. The project applies various machine learning algorithms, including logistic models, tree-based methods, and boosting techniques such as **AdaBoost, Gradient Boosting, and XGBoost**, while carefully addressing and preventing data leakage using proper train–test split pipelines. Model performance is evaluated using metrics like ROC-AUC, PR-AUC, Recall, and F1-Score to enable a structured comparison and determine the most effective approach for customer churn prediction.

---

## 🧩 Project Workflow Overview

This project follows an industry-standard ML pipeline:

- **Python (Pandas / Numpy)** → Data cleaning & preprocessing  
- **Data Leakage Prevention** → Ensuring transformations & resampling are applied only on training data using pipelines  
- **Imbalance Handling** → SMOTE, ADASYN, Undersampling, Class Weights  
- **Model Training** → Trained 6 different ML algorithms + sampling variants  
- **Model Evaluation** → Recall, F1, ROC-AUC, PR-AUC metrics  
- **Final Model Selection** → XGBoost with highest churn detection capability  

---

## 🧠 Model Selection Rationale & Challenges

This project focuses on a binary classification problem where the objective is to predict whether a customer will stay with the bank or churn (leave). I began by experimenting with three base models — **Logistic Regression**, **Decision Tree**, and **Random Forest**. Below is a detailed explanation of the challenges encountered with each model and the workflow followed to understand and improve their performance.

---

# 🔹 Logistic Regression Family

### Logistic Regression

Initially, I began by training a Logistic Regression model and evaluating its performance using the classification report. The model showed excellent metrics (accuracy, precision, recall, F1-score) for class 0 (non-churn), but performed very poorly for class 1 (churn). This clearly indicated that the model was biased toward predicting the majority class.

#### 📊 Classification Report (Logistic Regression)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.83 | 0.61 |
| Recall | 0.96 | 0.21 |
| F1-score | 0.89 | 0.32 |
| Support | 2389 | 611 |

It is clear from the above table that Class 0 significantly outperforms Class 1 across all metrics. This indicates that the model is biased toward the majority class and struggles to correctly identify churned customers due to the dataset’s imbalance.

To verify this, I examined the distribution of the target variable `Exited`, which showed a clear imbalance:
`y.value_counts()`
| Exited (y) | Count |
|------------|-------|
| 0 (Not churn) | 7960 |
| 1 (Churn) | 2037 |

---

### Logistic Regression + Undersampling

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

### Logistic Regression + SMOTE

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

### Logistic Regression + ADASYN

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

### Logistic Regression + Class Weighting

In this approach, instead of modifying the dataset, class weights were applied to penalize misclassification of the minority class. This instructs the model to treat churn cases (class 1) as more important during training.<br>
`steps = [("preprocess", preprocessor),`  
`         ("logistic_regression", LogisticRegression(random_state=42, class_weight='balanced'))]`

#### 📊 Classification Report (Logistic Regression + Class Weights)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.91 | 0.39 |
| Recall | 0.72 | 0.71 |
| F1-score | 0.80 | 0.51 |
| Support | 2389 | 611 |

This method performs similarly to SMOTE and ADASYN in terms of Class 1 recall, but without adding synthetic samples or removing original data. It provides a good balance between recall and data integrity, although precision for Class 1 remains moderate due to unavoidable false alarms.

---

### 🧩 Conclusion for Logistic Regression

Although Logistic Regression provides a simple and interpretable baseline model, its performance on this dataset remains limited due to the nonlinear relationships and strong class imbalance present in the target variable. Even after applying various imbalance handling techniques — such as undersampling, SMOTE, ADASYN, and class weighting — the model still struggled to consistently and reliably identify churners.

Therefore, Logistic Regression is not suitable as the final model choice for this problem, especially when more advanced and robust models like tree-based and boosting algorithms (e.g., Decision Tree, Random Forest, Gradient Boosting, XGBoost) can capture complex feature interactions and handle imbalance more effectively.

---

# 🔹 Decision Tree Family

### Decision Tree

A Decision Tree is a flowchart-like structure used for decision making and classification. Unlike Logistic Regression, Decision Trees can handle non-linear relationships between features. However, they also have a tendency to **overfit**, especially when the tree grows too deep and learns noise rather than patterns, leading to **low bias and high variance**.

Since the dataset is large and contains complex relationships, I did not manually control tree depth using pruning or `max_depth`. Instead, I performed **Hyperparameter Tuning** to optimize the model. Hyperparameters are parameters set **before** training that influence how the model learns.

To achieve this, I used `GridSearchCV`, which helps in finding the best combination of hyperparameters using cross-validation.

The parameter grid used:

`param_grid = {`  
`    'decision_tree__criterion': ['gini', 'entropy'],`  
`    'decision_tree__splitter': ['best', 'random'],`  
`    'decision_tree__max_depth': [None, 10, 20, 30],`  
`    'decision_tree__min_samples_split': [2, 5, 10],`  
`    'decision_tree__min_samples_leaf': [1, 2, 4]`  
`}`

Scoring metrics used during grid search:

`scoring = {`  
`    'accuracy' : 'accuracy',`  
`    'precision' : 'precision',`  
`    'recall' : 'recall',`  
`    'f1' : 'f1'`  
`}`

Grid search execution:

`grid_search = GridSearchCV(estimator = pipe,`  
`                           param_grid = param_grid,`  
`                           cv = 5,`  
`                           scoring = scoring,`  
`                           refit = 'recall'`  
`)`

#### 📊 Classification Report (Decision Tree)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.88 | 0.50 |
| Recall | 0.86 | 0.53 |
| F1-score | 0.87 | 0.51 |
| Support | 2389 | 611 |

It is important to note that this base Decision Tree model was trained on the original imbalanced dataset, without any resampling or class weighting applied. This imbalance contributed to the weaker performance on the churn class, reinforcing the need for imbalance handling techniques and more robust ensemble models.

---

### Decision Tree + Undersampling

Since undersampling was already demonstrated previously, the same approach was applied here using Decision Tree to check its impact on performance.

#### 📊 Classification Report (Decision Tree + Undersampling)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.92 | 0.45 |
| Recall | 0.77 | 0.74 |
| F1-score | 0.84 | 0.56 |
| Support | 2389 | 611 |

With undersampling, the Decision Tree also shows a strong boost in recall for Class 1 (churn), improving from 0.53 (without sampling) to 0.74. However, the precision decreases for Class 1 due to more false positives, similar to the effect observed in the logistic regression undersampling approach.

---

### Decision Tree + SMOTE

Applying SMOTE to Decision Tree allowed the model to train on a synthetically balanced dataset without losing original class 0 data. This helps the tree better capture churn patterns that were previously underrepresented.

#### 📊 Classification Report (Decision Tree + SMOTE)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.91 | 0.40 |
| Recall | 0.72 | 0.71 |
| F1-score | 0.81 | 0.51 |
| Support | 2389 | 611 |

This configuration significantly improves the detection of churners while maintaining the benefit of using all original training data. Similar to the Logistic Regression + SMOTE case, recall for Class 1 improves, but precision decreases, indicating an increase in false positives for churn prediction.

---

### Decision Tree + ADASYN

Using ADASYN with Decision Tree further emphasizes samples that are harder to classify by generating synthetic points in regions where the minority class is underrepresented. This helps the model better generalize on challenging churn cases.

#### 📊 Classification Report (Decision Tree + ADASYN)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.92 | 0.46 |
| Recall | 0.78 | 0.74 |
| F1-score | 0.84 | 0.57 |
| Support | 2389 | 611 |

With ADASYN, the Decision Tree achieves a strong recall of 0.74 for the churn class, and shows better precision and F1 performance compared to SMOTE. This indicates that the model benefits from ADASYN’s adaptive oversampling approach in learning difficult churn patterns more effectively.

---

### Decision Tree + Class Weighting

Instead of modifying the dataset, class weights were applied to penalize misclassification of churn cases. This makes the model assign greater importance to minority class samples during training.

#### 📊 Classification Report (Decision Tree + Class Weighting)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.92 | 0.46 |
| Recall | 0.78 | 0.73 |
| F1-score | 0.84 | 0.56 |
| Support | 2389 | 611 |

This approach delivers performance similar to ADASYN, achieving strong recall for churn cases while avoiding synthetic data generation. Class weighting gives the model a balanced focus without altering the dataset, maintaining data integrity with competitive prediction capability.

---

### 🧩 Conclusion for Decision Tree

While the Decision Tree demonstrates meaningful improvement over Logistic Regression and captures nonlinear patterns, it is still prone to overfitting and limited generalization. Random Forest, being an ensemble of multiple decision trees, addresses this limitation by reducing variance and improving overall model stability. Therefore, Decision Tree alone is not selected as the final model choice.

---

# 🔹 Random Forest Family

### Random Forest

Random Forest is an ensemble learning method that builds multiple Decision Trees and combines their outcomes to produce more stable and accurate predictions. It follows the **bagging (bootstrap aggregating)** strategy, where multiple trees are trained in parallel on different random subsets of the data. 

This significantly reduces overfitting compared to a single Decision Tree by averaging predictions across many trees. In classification problems, the final output is determined through **majority voting**, making Random Forest more robust and reliable.

To optimize the Random Forest model, I performed **Hyperparameter Tuning** using `GridSearchCV`, which finds the best parameter combination through exhaustive cross-validation.

The parameter grid used:

`param_grid = {`  
`    'random_forest__criterion': ['gini', 'entropy'],`  
`    'random_forest__max_depth': [3, 5, 7, 10, 15, 20, None],`  
`    'random_forest__min_samples_split': [2, 5, 10],`  
`    'random_forest__min_samples_leaf': [1, 2, 4]`  
`}`

Scoring metrics used during grid search:

`scoring = {`  
`    'accuracy' : 'accuracy',`  
`    'precision' : 'precision',`  
`    'recall' : 'recall',`  
`    'f1' : 'f1'`  
`}`

Grid search execution:

`grid_search = GridSearchCV(estimator=pipe,`  
`                           param_grid = param_grid,`  
`                           cv = 5,`  
`                           scoring = scoring,`  
`                           refit = 'recall'`  
`)`

#### 📊 Classification Report (Random Forest)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.88 | 0.75 |
| Recall | 0.96 | 0.49 |
| F1-score | 0.92 | 0.59 |
| Support | 2389 | 611 |

It is important to note that these results are from the base Random Forest model trained directly on the original imbalanced dataset, without the use of any resampling or class weighting methods.
Random Forest clearly outperforms the Decision Tree, achieving a much better balance between recall and precision while reducing overfitting due to its ensemble nature. The model captures churn cases more effectively and demonstrates stronger generalization performance.

---

### Random Forest + Undersampling

Undersampling was applied to Random Forest to balance the dataset by reducing the majority class. Since undersampling was already explained earlier, the same concept is used here without repeating the full theory.

#### 📊 Classification Report (Random Forest + Undersampling)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.92 | 0.50 |
| Recall | 0.81 | 0.74 |
| F1-score | 0.86 | 0.59 |
| Support | 2389 | 611 |

Using undersampling, Random Forest achieves a strong recall of **0.74** for churn cases while maintaining solid overall performance. Although precision for the churn class decreases due to more false positives, the model becomes significantly better at identifying customers likely to leave.

---

### Random Forest + SMOTE

Applying SMOTE with Random Forest generates synthetic samples for the minority class, allowing the model to learn churn patterns more effectively without losing any original data.

#### 📊 Classification Report (Random Forest + SMOTE)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.92 | 0.51 |
| Recall | 0.82 | 0.72 |
| F1-score | 0.87 | 0.60 |
| Support | 2389 | 611 |

SMOTE helps Random Forest maintain high recall for churners (0.72) while using the full dataset with synthetic samples. This leads to a balanced performance, and a strong ability to detect churn customers more reliably compared to the base model.

---

### Random Forest + ADASYN

ADASYN emphasizes difficult-to-classify minority samples by generating more synthetic data in regions where churners are surrounded by majority class samples. This helps Random Forest focus on complex boundary cases.

#### 📊 Classification Report (Random Forest + ADASYN)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.93 | 0.46 |
| Recall | 0.78 | 0.76 |
| F1-score | 0.84 | 0.57 |
| Support | 2389 | 611 |

ADASYN improves the recall for churners to **0.76**, the highest among RF techniques so far. However, precision drops due to more aggressive churn predictions. This shows that ADASYN helps identify more churners, but increases false positives slightly.

---

### Random Forest + Class Weighting

Here, instead of modifying the dataset, class weights were applied to penalize misclassification of the churn class — giving more importance to class 1 during training.

#### 📊 Classification Report (Random Forest + Class Weighting)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.93 | 0.52 |
| Recall | 0.82 | 0.74 |
| F1-score | 0.87 | 0.61 |
| Support | 2389 | 611 |

Class weighting produces almost the same performance as SMOTE, achieving a good recall of 0.74 for churners without generating synthetic data or removing samples. This makes it an efficient approach that maintains dataset integrity while improving minority class detection.

---

### 🧩 Conclusion for Random Forest

Random Forest demonstrates a strong improvement over the single Decision Tree model by reducing overfitting, improving overall stability, and achieving high recall for the churn class, especially with class weighting and ADASYN. However, while it performs reliably and provides balanced results, there is still room for improvement in capturing minority-class patterns. This motivated the exploration of more powerful sequential ensemble techniques such as Gradient Boosting and XGBoost, which can further optimize churn detection.

---

# 🔹 Boosting Techniques Overview

Boosting is an ensemble method where models (weak learners) are built sequentially — each subsequent model focuses on correcting the errors made by the previous ones. Individually, these weak learners may not perform very well, but when combined, they form a strong and highly accurate predictive model.

In this project, I implemented three boosting algorithms:
- **AdaBoost (Adaptive Boosting)**
- **Gradient Boosting**
- **XGBoost (Extreme Gradient Boosting)**

Below, I provide the detailed explanation and performance analysis of each boosting method used in this project.

---

## AdaBoost Section

### AdaBoost (Adaptive Boosting)

AdaBoost works by iteratively training weak learners (typically shallow decision trees) and assigning higher weights to the samples that were misclassified in previous rounds. This forces subsequent models to focus increasingly on the difficult cases. Over successive iterations, these weighted weak learners combine to form a strong classifier.

To tune AdaBoost, I used `GridSearchCV` with the following hyperparameter grid:

`param_grid = {`  
`    'ada_boost__n_estimators': [50, 100, 200, 300],`  
`    'ada_boost__learning_rate': [0.001, 0.01, 0.1, 1.0]`  
`}`

Performance metrics used during grid search:

`scoring = {`  
`    'accuracy': 'accuracy',`  
`    'precision': 'precision',`  
`    'recall': 'recall',`  
`    'f1': 'f1'`  
`}`

Grid search execution:

`grid_search = GridSearchCV(estimator = pipe,`  
`                           param_grid = param_grid,`  
`                           scoring = scoring,`  
`                           refit = 'recall',`  
`                           cv = 5,`  
`)`

#### 📊 Classification Report (AdaBoost)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.88 | 0.75 |
| Recall | 0.96 | 0.49 |
| F1-score | 0.92 | 0.60 |
| Support | 2389 | 611 |

AdaBoost achieves very high precision for the churn class, meaning that when it predicts churn, it is often correct. However, its recall for churn (0.49) is significantly lower than other models, indicating that it fails to identify a substantial number of customers who are likely to leave.

---

### AdaBoost (Varying Base Estimator Depth)

Since AdaBoost commonly performs best with very shallow decision trees (decision stumps), I experimented with varying the depth of the base estimator from 1 to 6. The observation from this experiment was that as the depth increased, the model started overfitting — learning the majority class too strongly and failing to generalize to the minority class (churners). This resulted in consistently weaker performance for class 1 with deeper trees.

For a detailed view of how metrics changed across different depths, refer to the model comparison report.

---

### AdaBoost + SMOTE

Since AdaBoost on its own does not perform well on imbalanced data, applying SMOTE helped balance the dataset by generating synthetic samples for the churn class, resulting in improved performance for Class 1.

#### 📊 Classification Report (AdaBoost + SMOTE)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.92 | 0.52 |
| Recall | 0.82 | 0.74 |
| F1-score | 0.87 | 0.61 |
| Support | 2389 | 611 |

This configuration significantly boosts recall for churners (0.74) compared to base AdaBoost (0.49), demonstrating that SMOTE helps AdaBoost better capture minority class patterns. However, the precision for churn remains moderate due to an increase in false positives, which is an expected trade-off when using oversampling techniques.

---

### AdaBoost + ADASYN

Using ADASYN emphasizes difficult-to-learn minority samples by generating synthetic churn examples in regions where class 1 is underrepresented. This allows AdaBoost to focus more on challenging churn cases during training.

#### 📊 Classification Report (AdaBoost + ADASYN)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.93 | 0.46 |
| Recall | 0.77 | 0.76 |
| F1-score | 0.84 | 0.57 |
| Support | 2389 | 611 |

Compared to SMOTE, ADASYN provides a similar improvement in churn recall (0.76), but slightly lower precision for the minority class. This indicates that ADASYN helps AdaBoost detect more churners, but also results in increased false churn predictions.

---

### 🧩 Conclusion for AdaBoost

AdaBoost demonstrated strong precision for churn predictions and benefits from its focus on misclassified samples. However, in its base form, it struggled with recall for the churn class due to dataset imbalance. Even after applying SMOTE and ADASYN, while recall improved significantly, the increase in false positives reduced its overall reliability. Therefore, although AdaBoost is effective, it does not provide the most optimal balance between recall, precision, and generalization for this churn prediction task.

---

## Gradient Boosting Section

### Gradient Boosting

Gradient Boosting builds trees sequentially, where each new tree attempts to correct the residual errors (the difference between actual and predicted values) made by the previous model. This method is theoretically more powerful than AdaBoost since it optimizes using gradients rather than reweighting samples.

#### 📊 Classification Report (Gradient Boosting)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.88 | 0.61 |
| Recall | 0.92 | 0.51 |
| F1-score | 0.90 | 0.56 |
| Support | 2389 | 611 |

While Gradient Boosting delivers high accuracy and strong performance on the majority class, the recall for churners remains relatively weak due to class imbalance. This indicates that without resampling or weighting strategies, Gradient Boosting gravitates toward learning the dominant class more effectively and struggles to capture minority-class churn patterns.

---

### Gradient Boosting + SMOTE

Applying SMOTE with Gradient Boosting helped balance the dataset by generating synthetic samples for minority class 1, allowing the model to better learn churn-related patterns.

#### 📊 Classification Report (Gradient Boosting + SMOTE)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.92 | 0.51 |
| Recall | 0.82 | 0.73 |
| F1-score | 0.87 | 0.60 |
| Support | 2389 | 611 |

With SMOTE, Gradient Boosting shows a significant improvement in recall for churn (0.73 vs. 0.51 in base model). This indicates that oversampling helps the model identify a larger proportion of churners. However, precision for churn remains moderate, and performance still trails behind Random Forest and XGBoost approaches.

---

### Gradient Boosting + ADASYN

#### 📊 Classification Report (Gradient Boosting + ADASYN)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.00 | 0.20 |
| Recall | 0.00 | 1.00 |
| F1-score | 0.00 | 0.34 |
| Support | 2389 | 611 |

In this scenario, the model predicted **every single sample** as churn (Class 1), resulting in:
- Recall = **1.00** for churn (it found all churners)
- But precision = **0.20** (because most of those predictions were incorrect)
- And recall = **0.00** for non-churn (because it didn’t predict a single class 0 instance)

This led to a total accuracy of only **20%**, meaning 80% of the predictions were wrong.

When ADASYN oversampled the minority class, it generated many synthetic churn points, especially around the decision boundaries. This shifted the dataset distribution heavily toward class 1. 

Gradient Boosting optimizes residual errors during training, and due to this imbalance, the residual gradient updates became biased toward correctly predicting churn. Eventually, the model converged to a trivial solution of predicting everything as churn to minimize loss — resulting in a collapsed classifier.

---

### 🧩 Conclusion for Gradient Boosting

Gradient Boosting demonstrated strong capability in modeling complex relationships and delivered high accuracy for the majority class. However, its performance on the minority churn class remained limited in the base form due to imbalance sensitivity. Even after applying SMOTE and ADASYN, the model either improved recall modestly or, in the case of ADASYN, collapsed into predicting all samples as churn. Overall, Gradient Boosting did not provide the most stable or reliable performance for churn detection when compared to Random Forest and XGBoost.

---

## XGBoost Section

### XGBoost (Extreme Gradient Boosting)

XGBoost is an optimized and regularized version of Gradient Boosting that incorporates advanced features such as:
- L1 and L2 regularization (reduces overfitting)
- optimized tree splitting
- handling sparse data efficiently
- parallelized computation
- improved missing value handling

It is widely known for being one of the most powerful models in tabular structured data.

#### 📊 Classification Report (XGBoost)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.93 | 0.50 |
| Recall | 0.80 | 0.76 |
| F1-score | 0.86 | 0.60 |
| Support | 2389 | 611 |

XGBoost achieves a strong recall of **0.76** for churn cases — significantly better than the base models. This demonstrates that XGBoost is able to capture more minority class patterns while still maintaining high precision for non-churn predictions.

---

### XGBoost + SMOTE

Applying SMOTE with XGBoost generates synthetic samples for churn cases and helps reduce the class imbalance while still leveraging XGBoost’s strong regularization and tree optimization capabilities.

#### 📊 Classification Report (XGBoost + SMOTE)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.93 | 0.53 |
| Recall | 0.84 | 0.73 |
| F1-score | 0.88 | 0.62 |
| Support | 2389 | 611 |

XGBoost combined with SMOTE achieves stronger recall for churn (0.73) and improves overall model balance, resulting in higher accuracy (0.82). SMOTE helps XGBoost capture more churn cases than the base XGBoost, although it slightly impacts precision due to more aggressive positive predictions.

---

### XGBoost + ADASYN

Using ADASYN, additional synthetic minority samples were generated, especially for harder-to-classify churn cases. This allows XGBoost to learn more subtle churn patterns near the decision boundary.

#### 📊 Classification Report (XGBoost + ADASYN)

| Metric | Class 0 (Not churn) | Class 1 (Churn) |
|--------|-------------------|----------------|
| Precision | 0.93 | 0.47 |
| Recall | 0.78 | 0.76 |
| F1-score | 0.85 | 0.58 |
| Support | 2389 | 611 |

ADASYN increases the recall of churn detection to **0.76**, matching the trend seen with other oversampling techniques. However, precision for churn decreases to **0.47**, indicating that the model flags more customers as churn incorrectly. This is a typical trade-off seen when aggressively expanding the minority class using adaptive oversampling.

---

### 🧩 Conclusion for XGBoost

XGBoost emerged as the best-performing model across all experiments due to its ability to generalize well, handle complex nonlinear relationships, and maintain stability even under sampling-based imbalance corrections. It consistently delivered high recall for the churn class while preserving strong precision for non-churn predictions, making it a highly reliable model for identifying at-risk customers. Considering both performance metrics and real-world usability, XGBoost provides the most balanced and effective solution for churn prediction in this project.

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



