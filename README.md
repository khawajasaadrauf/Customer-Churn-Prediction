# Customer Churn Prediction

## Overview

This project focuses on predicting customer churn for a telecom company using machine learning techniques. Churn prediction is crucial for businesses to retain customers by understanding why they leave and taking proactive steps to prevent it. By analyzing customer data and building a predictive model, this project aims to identify customers who are likely to cancel their services.

## Tools and Libraries

- Programming Language: Python
- Data Manipulation: Pandas, NumPy
- Machine Learning: Scikit-learn
- Data Visualization: Matplotlib, Seaborn
- Development Environment: Jupyter Notebook

## Dataset

The dataset used in this project is the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn) from Kaggle. It contains information about customer demographics, account information, and service usage, along with whether or not the customer churned.

### Key Features:

- Demographics: Gender, age range, etc.
- Services: Type of services each customer has signed up for (internet, phone, streaming, etc.)
- Account Information: Contract type, payment method, tenure, monthly charges, etc.
- Churn: Whether the customer churned (target variable)

## Project Workflow

### 1. Data Exploration

We started by loading the dataset and conducting an exploratory data analysis (EDA) to understand the structure, quality, and distribution of the data. Key observations included:

- The dataset had 7,043 records and 21 columns.
- The target variable, `Churn`, was imbalanced, with about 26% of customers having churned.
- Key features impacting churn appeared to be `tenure`, `contract type`, `monthly charges`, and `total charges`.

### 2. Data Preprocessing

Data preprocessing steps included:

- Handling Missing Values: We identified and removed missing values to ensure data integrity.
- Encoding Categorical Variables: Categorical variables were converted to numerical format using one-hot encoding.
- Feature Scaling: Continuous features were standardized to improve the performance of the machine learning models.
- Train-Test Split: The dataset was split into training (80%) and testing (20%) sets to evaluate the model's performance.

### 3. Model Building

We used a Random Forest Classifier, a robust ensemble learning method, to predict customer churn. Random Forest was chosen for its ability to handle large datasets and model complex relationships without overfitting.

Model Training:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 4. Model Evaluation

The model was evaluated on the test data using several metrics:

- Confusion Matrix: A matrix showing true positives, true negatives, false positives, and false negatives.
- Classification Report: Including precision, recall, F1-score, and accuracy.

Results:

- Accuracy: 79%
- Precision: 72%
- Recall: 65%
- F1-Score: 68%

```python
from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
```

### 5. Feature Importance Visualization

We visualized the importance of features to understand which factors contribute most to customer churn.

```python
import pandas as pd

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.show()
```

Top Features Influencing Churn:

1. Tenure
2. Monthly Charges
3. Contract Type
4. Total Charges
5. Internet Service

These insights can guide targeted retention strategies, such as offering discounts to customers with short tenures or high monthly charges.

## Conclusion

The model achieved a good balance between precision and recall, making it a reliable tool for predicting customer churn. The feature importance analysis provided actionable insights that can help the company focus on high-risk customers and reduce churn rates.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. Set up the environment and install dependencies:
   ```bash
   python -m venv churn_env
   source churn_env/bin/activate  # On Windows use `churn_env\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Explore and run the code: Open `Customer_Churn_Prediction.ipynb` to view the analysis and model-building steps.

## Future Work

- Model Improvement: Experiment with other models like Gradient Boosting or XGBoost to improve accuracy.
- Hyperparameter Tuning: Fine-tune the Random Forest model parameters to achieve better performance.
- Advanced Analytics: Implement causal inference techniques to understand the impact of interventions on churn.
- Dashboarding: Create an interactive dashboard using PowerBI or Tableau to visualize churn predictions and insights in real-time.

## Acknowledgments

Thanks to Kaggle for providing the dataset and to the open-source community for the tools and libraries that made this project possible.
