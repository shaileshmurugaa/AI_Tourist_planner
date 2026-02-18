%%writefile app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the models, scaler, and rule engine
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
xgboost_model = joblib.load('xgboost_model.pkl')
apply_travel_rules = joblib.load('rule_engine.pkl')

# Let's get the column names from the trained X_train to ensure consistency
# Assuming X_train's columns represent the correct order and set of features expected by the models
expected_feature_columns = random_forest_model.feature_names_in_

# Function to preprocess incoming data
def preprocess_data(data):
    df_new = pd.DataFrame([data])

    if 'Start date' in df_new.columns:
        df_new['Start date'] = pd.to_datetime(df_new['Start date'])
        df_new['Travel_Month'] = df_new['Start date'].dt.month
    elif 'Travel_Month' not in df_new.columns:
        df_new['Travel_Month'] = 1 # Example default

    df_initial_features = pd.DataFrame([data])
    df_initial_features['Start date'] = pd.to_datetime(df_initial_features['Start date'])
    df_initial_features['End date'] = pd.to_datetime(df_initial_features['End date'])
    df_initial_features['Travel_Month'] = df_initial_features['Start date'].dt.month
    df_initial_features = df_initial_features.drop(columns=['Start date', 'End date', 'Trip ID', 'Traveler name'], errors='ignore')

    categorical_cols = [
        'Destination',
        'Traveler gender',
        'Traveler nationality',
        'Accommodation type',
        'Transportation type'
    ]
    for col in categorical_cols:
        if col not in df_initial_features.columns:
            df_initial_features[col] = np.nan

    df_features_ohe = pd.get_dummies(df_initial_features, columns=categorical_cols, drop_first=True)

    kmeans_feature_columns = scaler.feature_names_in_

    template_for_kmeans = pd.DataFrame(columns=kmeans_feature_columns)
    df_for_kmeans_input = pd.concat([template_for_kmeans, df_features_ohe], ignore_index=True, sort=False).fillna(0)
    df_for_kmeans_input = df_for_kmeans_input[kmeans_feature_columns]

    X_for_kmeans_scaled = scaler.transform(df_for_kmeans_input)

    new_cluster_label = kmeans_model.predict(X_for_kmeans_scaled)[0]

    new_cluster_df = pd.DataFrame(0, index=[0], columns=['Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3'])
    new_cluster_df[f'Cluster_{new_cluster_label}'] = 1

    original_feature_part_of_X_train = [col for col in expected_feature_columns if not col.startswith('Cluster_')]
    template_original_features = pd.DataFrame(columns=original_feature_part_of_X_train)

    df_all_features = pd.concat([template_original_features, df_features_ohe], ignore_index=True, sort=False).fillna(0)
    df_all_features = df_all_features[original_feature_part_of_X_train]

    for col in new_cluster_df.columns:
        df_all_features[col] = new_cluster_df[col].iloc[0]

    df_final_for_prediction = df_all_features[expected_feature_columns]

    return df_final_for_prediction, df_new.iloc[0]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        processed_data_for_models, original_data_row = preprocess_data(data)

        rf_pred = random_forest_model.predict(processed_data_for_models)[0]
        xgb_pred = xgboost_model.predict(processed_data_for_models)[0]

        adjusted_pred = apply_travel_rules(original_data_row, rf_pred, xgb_pred)

        return jsonify({
            'random_forest_prediction': float(rf_pred),
            'xgboost_prediction': float(xgb_pred),
            'final_adjusted_prediction': float(adjusted_pred)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
