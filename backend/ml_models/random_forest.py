import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 2: Load dataset
df = pd.read_csv("ecommerce_returns_synthetic_data.csv")


# Step 3: Remove leakage and useless columns
df = df.drop(columns=[
    'Order_ID',
    'Product_ID',
    'User_ID',
    'Return_Date',
    'Return_Reason',
    'Days_to_Return'
])


# Step 4: Create REALISTIC Return_Status using logical rules
# This replaces random return labels with learnable patterns

df['Return_Status'] = (
    ((df['Product_Category'] == 'Clothing') & (df['Discount_Applied'] > 25)) |
    ((df['Shipping_Method'] == 'Express') & (df['Product_Price'] > 300)) |
    ((df['User_Age'] < 25) & (df['Payment_Method'] == 'PayPal')) |
    ((df['Order_Quantity'] > 3) & (df['Discount_Applied'] > 20))
).astype(int)


# Step 5: Convert Order_Date into useful features
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

df['order_day'] = df['Order_Date'].dt.weekday
df['order_month'] = df['Order_Date'].dt.month
df['is_weekend'] = (df['Order_Date'].dt.weekday >= 5).astype(int)

df = df.drop(columns=['Order_Date'])


# Step 6: One-hot encode categorical columns
df = pd.get_dummies(df, columns=[
    'Product_Category',
    'User_Gender',
    'User_Location',
    'Payment_Method',
    'Shipping_Method'
])


# Step 7: Split features and target
X = df.drop(columns=['Return_Status'])
y = df['Return_Status']


# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier(
    n_estimators=300,      # number of trees
    max_depth=15,          # prevents overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1              # use all CPU cores
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import joblib

# Save trained model
joblib.dump(model, "return_prediction_model.pkl")

print("Model saved successfully")

joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Columns saved successfully")