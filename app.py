from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
@app.route("/")
def home():
    return "API Running!"

# Load models
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
xgboost_model = joblib.load('xgboost_model.pkl')
apply_travel_rules = joblib.load('rule_engine.pkl')

# Expected feature columns
expected_feature_columns = random_forest_model.feature_names_in_

# Home route (for Render health check)
@app.route('/')
def home():
    return "AI Climate-Smart Tourist Planner API is running!"

# Preprocessing function
def preprocess_data(data):
    df_new = pd.DataFrame([data])

    # Convert dates
    if 'Start date' in df_new.columns:
        df_new['Start date'] = pd.to_datetime(df_new['Start date'])
        df_new['Travel_Month'] = df_new['Start date'].dt.month
    elif 'Travel_Month' not in df_new.columns:
        df_new['Travel_Month'] = 1

    df_initial = pd.DataFrame([data])
    df_initial['Start date'] = pd.to_datetime(df_initial['Start date'])
    df_initial['End date'] = pd.to_datetime(df_initial['End date'])
    df_initial['Travel_Month'] = df_initial['Start date'].dt.month

    df_initial = df_initial.drop(columns=['Start date', 'End date', 'Trip ID', 'Traveler name'], errors='ignore')

    categorical_cols = [
        'Destination',
        'Traveler gender',
        'Traveler nationality',
        'Accommodation type',
        'Transportation type'
    ]

    for col in categorical_cols:
        if col not in df_initial.columns:
            df_initial[col] = np.nan

    df_ohe = pd.get_dummies(df_initial, columns=categorical_cols, drop_first=True)

    # KMeans input
    kmeans_features = scaler.feature_names_in_
    template = pd.DataFrame(columns=kmeans_features)

    df_kmeans = pd.concat([template, df_ohe], ignore_index=True, sort=False).fillna(0)
    df_kmeans = df_kmeans[kmeans_features]

    X_scaled = scaler.transform(df_kmeans)
    cluster_label = kmeans_model.predict(X_scaled)[0]

    cluster_df = pd.DataFrame(0, index=[0], columns=['Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3'])
    cluster_df[f'Cluster_{cluster_label}'] = 1

    original_cols = [col for col in expected_feature_columns if not col.startswith('Cluster_')]
    template_orig = pd.DataFrame(columns=original_cols)

    df_all = pd.concat([template_orig, df_ohe], ignore_index=True, sort=False).fillna(0)
    df_all = df_all[original_cols]

    for col in cluster_df.columns:
        df_all[col] = cluster_df[col].iloc[0]

    final_df = df_all[expected_feature_columns]

    return final_df, df_new.iloc[0]

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        processed_data, original_row = preprocess_data(data)

        rf_pred = random_forest_model.predict(processed_data)[0]
        xgb_pred = xgboost_model.predict(processed_data)[0]

        final_pred = apply_travel_rules(original_row, rf_pred, xgb_pred)

        return jsonify({
            'random_forest_prediction': float(rf_pred),
            'xgboost_prediction': float(xgb_pred),
            'final_adjusted_prediction': float(final_pred)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
