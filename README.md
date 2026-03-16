# 🏠 California House Price Prediction using XGBoost

## 📌 Project Overview

This project builds a **Machine Learning regression model** to predict house prices in California districts using the **California Housing Dataset**.
The model is trained using **XGBoost Regressor**, one of the most powerful gradient boosting algorithms widely used in data science competitions and real-world applications.

The goal of this project is to demonstrate the complete **Machine Learning workflow**, including:

* Data loading
* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature correlation analysis
* Data splitting
* Model training
* Model evaluation

---

# 📊 Dataset Information

The dataset used in this project is the **California Housing Dataset**, available in `sklearn.datasets`.

It contains information collected from the **1990 U.S. Census**.

### Dataset Characteristics

| Property            | Value              |
| ------------------- | ------------------ |
| Number of Instances | 20,640             |
| Number of Features  | 8                  |
| Target Variable     | Median House Value |
| Data Type           | Numerical          |
| Missing Values      | None               |

The target variable represents the **median house value in California districts**, expressed in **hundreds of thousands of dollars**.

Example:

* Value **2.5** → $250,000
* Value **4.0** → $400,000

---

# 📂 Features Description

| Feature    | Description                                 |
| ---------- | ------------------------------------------- |
| MedInc     | Median income of households in the district |
| HouseAge   | Median age of houses in the district        |
| AveRooms   | Average number of rooms per household       |
| AveBedrms  | Average number of bedrooms per household    |
| Population | Total population in the district            |
| AveOccup   | Average number of household members         |
| Latitude   | Latitude coordinate of the district         |
| Longitude  | Longitude coordinate of the district        |

---

# ⚙️ Technologies & Libraries

This project uses the following Python libraries:

* **NumPy** → numerical computations
* **Pandas** → data manipulation and analysis
* **Matplotlib** → data visualization
* **Seaborn** → statistical visualization
* **Scikit-learn** → machine learning utilities
* **XGBoost** → gradient boosting regression model

Install dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

# 🧠 Machine Learning Workflow

## 1️⃣ Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
```

---

# 2️⃣ Load Dataset

```python
from sklearn.datasets import fetch_california_housing

house_price_dataset = fetch_california_housing()
```

---

# 3️⃣ Convert Dataset to DataFrame

```python
house_price_dataframe = pd.DataFrame(
    house_price_dataset.data,
    columns = house_price_dataset.feature_names
)

house_price_dataframe['price'] = house_price_dataset.target
```

---

# 4️⃣ Data Exploration

### Check Missing Values

```python
house_price_dataframe.isnull().sum()
```

Result:
No missing values were found in the dataset.

---

### Statistical Summary

```python
house_price_dataframe.describe()
```

Provides statistical metrics including:

* mean
* standard deviation
* minimum and maximum values
* quartiles

---

# 📈 Correlation Analysis

To understand relationships between features and the target variable, a **correlation matrix** is generated.

```python
correlation = house_price_dataframe.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation, annot=True, fmt='.1f', square=True)
```

This visualization helps identify features that strongly influence house prices.

Typically:

* **Median Income (MedInc)** has the strongest positive correlation with house prices.

---

# 🔀 Train-Test Split

The dataset is divided into **training and testing sets**.

```python
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)
```

Split ratio:

* **80% Training Data**
* **20% Testing Data**

---

# 🤖 Model Training

The model used is **XGBoost Regressor**.

```python
model = XGBRegressor()
model.fit(X_train, Y_train)
```

XGBoost builds an ensemble of decision trees using **gradient boosting**, which improves prediction accuracy.

---

# 📉 Model Evaluation

## Training Data Performance

```python
training_data_prediction = model.predict(X_train)

r2_train = metrics.r2_score(Y_train, training_data_prediction)
mae_train = metrics.mean_absolute_error(Y_train, training_data_prediction)
```

Results:

| Metric   | Value |
| -------- | ----- |
| R² Score | 0.94  |
| MAE      | 0.19  |

---

## Test Data Performance

```python
test_data_prediction = model.predict(X_test)

r2_test = metrics.r2_score(Y_test, test_data_prediction)
mae_test = metrics.mean_absolute_error(Y_test, test_data_prediction)
```

Results:

| Metric   | Value |
| -------- | ----- |
| R² Score | 0.83  |
| MAE      | 0.31  |

---

# 📊 Interpretation of Results

* **R² = 0.83** means the model explains **83% of the variance** in housing prices.
* **MAE ≈ 0.31** means the average prediction error is around **$31,000**.

This is a strong result considering the dataset only includes **8 features**.

---

# 🚀 Possible Improvements

Future improvements could include:

* Hyperparameter tuning
* Feature engineering
* Removing extreme outliers
* Using cross-validation
* Trying additional models such as:

  * Random Forest
  * Gradient Boosting
  * LightGBM

---

# 📌 Project Structure

```
House-Price-Prediction
│
├── House Price Prediction.ipynb
└── README.md
```

---

# 📚 References

* Scikit-learn Documentation
* XGBoost Documentation
* California Housing Dataset

---

# 👨‍💻 Author

**Machine Learning Project**

California Housing Price Prediction using **XGBoost Regression**.

---

⭐ If you found this project helpful, feel free to **star the repository**!
