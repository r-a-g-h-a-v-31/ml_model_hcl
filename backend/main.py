import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

pipeline = joblib.load(os.path.join(BASE_DIR, "ml_models", "return_prediction_model.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "ml_models", "model_columns.pkl"))

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
    app.run(debug=True, port=5000, use_reloader=False)
