# Titanic Dataset Analysis

This code performs a machine learning task to predict the survival outcome of passengers on the Titanic. It uses a Random Forest classifier to make the predictions based on several features of the passengers.

<!-- ## Data Preprocessing

1. Handling Missing Values:
   - The code first calculates the median age from the training dataset (`df_train`) and assigns it to the variable `age_median_tr`.
   - Similarly, it calculates the median age from the test dataset (`df_test`) and assigns it to the variable `age_median_ts`.
   - The missing values in the 'Age' column of both datasets are then replaced with their respective median values using the `replace` function.

2. Removing Columns:
   - The code removes the 'Cabin' column from both `df_train` and `df_test` datasets using the `drop` function.
   - It also removes the 'Embarked' column from both datasets using the `drop` function.

## Exploratory Analysis

The code performs a closer analysis of the 'Sex' attribute to understand the survival ratio of different genders on the Titanic.

- Women:
  - It selects the rows where 'Sex' is equal to 'female' from `df_train` using boolean indexing and assigns it to the variable `women`.
  - The code calculates the ratio of survived women by summing the values of the 'Survived' column for women and dividing it by the total number of women.
  - The calculated ratio is assigned to the variable `w_ratio`.

- Men:
  - It selects the rows where 'Sex' is equal to 'male' from `df_train` using boolean indexing and assigns it to the variable `men`.
  - The code calculates the ratio of survived men by summing the values of the 'Survived' column for men and dividing it by the total number of men.
  - The calculated ratio is assigned to the variable `m_ratio`. -->

## Machine Learning Model

The code builds a Random Forest classifier to predict the survival outcome based on the selected features.

1. Data Preparation:
   - The code assigns the 'Survived' column from `df_train` to the variable `y` as the target variable.
   - It selects the features ['Pclass', 'Sex', 'SibSp', 'Parch'] from `df_train` and encodes them using one-hot encoding with the `pd.get_dummies` function. The encoded features are assigned to the variable `X`.
   - Similarly, the code encodes the features from `df_test` using the same one-hot encoding scheme and assigns them to `X_test`.

2. Model Training:
   - The code initializes a Random Forest classifier with 100 trees, a maximum depth of 5, and a random state of 1. The classifier is assigned to the variable `model`.
   - It fits the classifier on the training data `X` and the target variable `y` using the `fit` function.

3. Making Predictions:
   - The code uses the trained model to make predictions on the test data `X_test` using the `predict` function. The predictions are assigned to the variable `predictions`.

4. Creating Output:
   - The code creates a DataFrame called `output` with the columns 'PassengerId' and 'Survived'.
   - It populates the 'PassengerId' column with the values from the 'PassengerId' column of `df_test` and the 'Survived' column with the predicted values from `predictions`.
   - The `output` DataFrame is then saved to a CSV file named 'submission.csv' using the `to_csv` function with `index=False` to exclude the index column.

## Prerequisites

To run this code, make sure you have the following installed:

- Python (version 3 or above)
- pandas library
- scikit-learn library

## Usage

1. Import the required libraries:
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
```

2. Read the training and test data files:
```python
df_train = pd.read_csv("./titanic/train.csv")
df_test = pd.read_csv("./titanic/test.csv")
```

3. Perform data preprocessing and exploratory analysis as described in the code.

4. Build and train the Random Forest classifier:
```python
y = df_train["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
```

5. Make predictions on the test data:
```python
predictions = model.predict(X_test)
```

6. Create the output DataFrame and save it to a CSV file:
```python
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
```

Feel free to modify the code and explore the dataset further using pandas and scikit-learn.
<!-- 
## License

This code is released under the [MIT License](https://opensource.org/licenses/MIT). -->
