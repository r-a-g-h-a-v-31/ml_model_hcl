import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

pipeline = None
feature_columns = None


def train_model():
    global pipeline, feature_columns

    df = pd.read_csv(os.path.join(BASE_DIR, "ml_models", "ecommerce_returns_synthetic_data.csv"))

    df = df.drop(columns=[
        'Order_ID', 'Product_ID', 'User_ID',
        'Return_Date', 'Return_Reason', 'Days_to_Return'
    ])

    df['Return_Status'] = (
        ((df['Product_Category'] == 'Clothing') & (df['Discount_Applied'] > 25)) |
        ((df['Shipping_Method'] == 'Express') & (df['Product_Price'] > 300)) |
        ((df['User_Age'] < 25) & (df['Payment_Method'] == 'PayPal')) |
        ((df['Order_Quantity'] > 3) & (df['Discount_Applied'] > 20))
    ).astype(int)

    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['order_day'] = df['Order_Date'].dt.weekday
    df['order_month'] = df['Order_Date'].dt.month
    df['is_weekend'] = (df['Order_Date'].dt.weekday >= 5).astype(int)
    df = df.drop(columns=['Order_Date'])

    df = pd.get_dummies(df, columns=[
        'Product_Category', 'User_Gender', 'User_Location',
        'Payment_Method', 'Shipping_Method'
    ])

    X = df.drop(columns=['Return_Status'])
    y = df['Return_Status']

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000, solver='liblinear'))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.4f}")


def prepare_input(data):
    input_dict = {}

    for col in feature_columns:
        input_dict[col] = 0

    input_dict['Product_Price'] = float(data['product_price'])
    input_dict['Order_Quantity'] = int(data['order_quantity'])
    input_dict['User_Age'] = int(data['user_age'])
    input_dict['Discount_Applied'] = float(data['discount_applied'])

    order_date = pd.to_datetime(data['order_date'])
    input_dict['order_day'] = order_date.weekday()
    input_dict['order_month'] = order_date.month
    input_dict['is_weekend'] = 1 if order_date.weekday() >= 5 else 0

    category_col = f"Product_Category_{data['product_category']}"
    if category_col in input_dict:
        input_dict[category_col] = 1

    gender_col = f"User_Gender_{data['user_gender']}"
    if gender_col in input_dict:
        input_dict[gender_col] = 1

    location_col = f"User_Location_{data['user_location']}"
    if location_col in input_dict:
        input_dict[location_col] = 1

    payment_col = f"Payment_Method_{data['payment_method']}"
    if payment_col in input_dict:
        input_dict[payment_col] = 1

    shipping_col = f"Shipping_Method_{data['shipping_method']}"
    if shipping_col in input_dict:
        input_dict[shipping_col] = 1

    input_df = pd.DataFrame([input_dict])
    return input_df[feature_columns]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = prepare_input(data)
        probability = pipeline.predict_proba(input_df)[0][1]
        prediction = "LIKELY TO BE RETURNED" if probability >= 0.5 else "NOT LIKELY TO BE RETURNED"

        return jsonify({
            'success': True,
            'return_probability': round(probability * 100, 2),
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    train_model()
    app.run(debug=True, port=5000, use_reloader=False)
