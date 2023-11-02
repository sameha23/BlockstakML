# Decision Tree & Naive Bayes Model
This code includes data preprocessing steps, such as handling missing values and standardizing numeric features. It then proceeds with building, training, and evaluating the Decision Tree and Naive Bayes models.



```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
```

This part of the code imports the necessary Python libraries and modules for data manipulation, machine learning, and evaluation. Here's a brief explanation of each library/module:

- `pandas`: A popular library for data manipulation and analysis.
- `sklearn.model_selection`: Provides functions for splitting the dataset into training and testing sets.
- `sklearn.preprocessing`: Contains tools for data preprocessing, such as scaling and encoding.
- `sklearn.tree`: Includes the DecisionTreeClassifier for decision tree-based classification.
- `sklearn.naive_bayes`: Contains the GaussianNB class for implementing a Naive Bayes classifier.
- `sklearn.metrics`: Provides functions for evaluating the models using metrics like accuracy, precision, recall, and F1-score.
- `sklearn.compose`: Includes the ColumnTransformer for applying different preprocessing to different subsets of the data.
- `sklearn.pipeline`: Helps create a pipeline of data preprocessing and modeling steps.

Now, let's move on to the next part of the code:

```python
# Load the dataset
data = pd.read_csv("your_dataset.csv")  # Replace with the actual dataset file path
```

This section loads the dataset from a CSV file. You should replace `"your_dataset.csv"` with the actual file path to your dataset.

```python
# Data preprocessing
# Handle missing values (if any)
data.dropna(inplace=True)
```

In the data preprocessing section, the code checks for and handles missing values. It uses the `dropna` method to remove rows with missing data. Depending on the dataset, you might want to consider more sophisticated methods for handling missing values, such as imputation.

```python
# Define the features and target variable
X = data.drop("y", axis=1)  # Features
y = data["y"]  # Target variable
```

Here, the code separates the dataset into features (X) and the target variable (y). The features consist of all columns except the target variable, which is labeled as "Y."

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

This section uses `train_test_split` from scikit-learn to split the data into training and testing sets. The training set is used to train the machine learning models, and the testing set is used to evaluate their performance.

```python
# Create a list of categorical and numeric columns
categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
numeric_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
```

The code defines two lists of column names: `categorical_columns` for categorical features and `numeric_columns` for numeric features.

```python
# Create transformers for preprocessing
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Standardize numeric features
])
```

In this section, transformers for preprocessing are defined for both categorical and numeric features. These transformers will be used in the data preprocessing step to convert categorical features using one-hot encoding and standardize numeric features.

```python
# Preprocess the data using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])
```

The `ColumnTransformer` is created to apply different preprocessing steps to different subsets of the data. It specifies that numeric features will be transformed using the `numeric_transformer`, and categorical features will be transformed using the `categorical_transformer`.

```python
# Create Decision Tree and Naive Bayes models
decision_tree_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

naive_bayes_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])
```

This section defines two machine learning models: a Decision Tree model and a Naive Bayes model. Each model is a pipeline that includes the data preprocessing step (ColumnTransformer) and the classifier (DecisionTreeClassifier for the Decision Tree model and GaussianNB for the Naive Bayes model).

```python
# Train the models
decision_tree_model.fit(X_train, y_train)
naive_bayes_model.fit(X_train, y_train)
```

The code trains both the Decision Tree and Naive Bayes models using the training data.

```python
# Make predictions
y_pred_decision_tree = decision_tree_model.predict(X_test)
y_pred_naive_bayes = naive_bayes_model.predict(X_test)
```

Predictions are made on the testing data using both the Decision Tree and Naive Bayes models.

```python
# Evaluate the models
decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree)
naive_bayes_accuracy = accuracy_score(y_test, y_pred_naive_bayes)

decision_tree_precision = precision_score(y_test, y_pred_decision_tree, average='binary', pos_label='yes')
naive_bayes_precision = precision_score(y_test, y_pred_naive_bayes, average='binary', pos_label='yes')

decision_tree_recall = recall_score(y_test, y_pred_decision_tree, average='binary', pos_label='yes')
naive_bayes_recall = recall_score(y_test, y_pred_naive_bayes, average='binary', pos_label='yes')

decision_tree_f1 = f1_score(y_test, y_pred_decision_tree, average='binary', pos_label='yes')
naive_bayes_f1 = f1_score(y_test, y_pred_naive_bayes, average='binary', pos_label='yes')
```

This part calculates various evaluation metrics for both models, including accuracy, precision, recall, and F1-score. These metrics assess how well each model performs in classifying subscribers and non-subscribers based on the input features.

```python
# Compare the results
print("Decision Tree Model:")
print(f"Accuracy: {decision_tree_accuracy}")
print(f"Precision: {decision_tree_precision}")
print(f"Recall: {decision_tree_recall}")
print(f"F1 Score: {decision_tree_f1}")

print("\nNaive Bayes Model:")
print(f"Accuracy: {naive_bayes_accuracy}")
print(f"Precision: {naive_bayes_precision}")
print(f"Recall: {naive_bayes_recall}")
print(f"F1 Score: {naive_bayes_f1}")
```

Finally, the code prints the results of the model evaluation, allowing us to compare the performance of the Decision Tree and Naive Bayes models. This comparison helps in understanding how well each model performs in classifying subscribers and non-subscribers and can guide decision-making in refining the telemarketing strategy.
