import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

# Load dataset
water_dataset = pd.read_csv("C:/Users/HP/Desktop/Regression/Water Quality project/water_potability.csv")

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
water_dataset[['ph', 'Sulfate', 'Trihalomethanes']] = imputer.fit_transform(water_dataset[['ph', 'Sulfate', 'Trihalomethanes']])

# Split data into features (X) and target variable (y)
X = water_dataset.drop('Potability', axis=1)
y = water_dataset['Potability']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict and evaluate
rf_y_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Classifier - Accuracy:", rf_accuracy)

# Save the trained model to a file
with open('watermodel.sav', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)
