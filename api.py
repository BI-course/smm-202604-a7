import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",
    "null",
]}})

MODEL_DIR = "./model"

regression_model = joblib.load(os.path.join(MODEL_DIR, "rent_price_model.joblib"))

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "logisticregression_classifier_for_employee_attrition.joblib")
)

clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "kmeans_retail_store_segmentation.joblib")
)
clustering_imputer = clustering_artifact["imputer"]
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "apriori_music_session_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]
track_lookup = arm_artifact["track_lookup"]

print("All the four artifacts have been loaded. The API is ready.")

RENT_CATEGORICAL = ["property_type", "neighborhood", "furnishing_status"]
RENT_NUMERIC = [
    "size_sqm", "num_bedrooms", "num_bathrooms", "distance_to_cbd_km",
    "service_charge_kes", "garage_size_sqm", "security_rating",
    "amenities_score", "nearby_schools_count", "has_garage",
]

ATTRITION_CATEGORICAL = ["department", "job_role_level", "education_level",
                         "performance_rating"]
ATTRITION_NUMERIC = [
    "years_since_last_promotion", "monthly_salary_kes",
    "overtime_hours_per_month", "job_satisfaction_score",
    "work_life_balance_score", "distance_from_home_km",
    "num_companies_worked_previously", "remote_work_days_per_week",
    "bonus_amount_kes",
]

STORE_NUMERIC = list(clustering_scaler.feature_names_in_)


def get_json_body():
    """Return the request's JSON body as a dict, or raise ValueError (-> 400)."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object "
                         "(send Content-Type: application/json).")
    return payload


def build_input_row(payload, categorical_fields, numeric_fields):
    """Check that every required field is present, then build a one-row
    DataFrame with the columns in the order the artifact was fitted on.

    JSON has no NaN and no int/float distinction: null arrives as None and
    12 arrives as an int. The imputers inside the pipelines were fitted on
    float columns where missing values are NaN, so numeric fields are cast to
    float (None -> NaN) before the DataFrame is built. Categorical fields are
    passed through untouched (None -> NaN so the imputer sees it as missing).
    """
    required = categorical_fields + numeric_fields
    missing = [f for f in required if f not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    row = {}
    for field in categorical_fields:
        value = payload[field]
        row[field] = np.nan if value is None else value
    for field in numeric_fields:
        value = payload[field]
        try:
            row[field] = np.nan if value is None else float(value)
        except (TypeError, ValueError):
            raise ValueError(f'"{field}" must be a number or null, got {value!r}.')
    return pd.DataFrame([row])[required]


@app.errorhandler(ValueError)
def handle_bad_input(error):
    # Raised by our own checks and by the pipelines themselves (for example an
    # OrdinalEncoder meeting a category it has never seen).
    return jsonify({"error": str(error)}), 400


# 1. REGRESSION -- predict the monthly rent (KES) of a property

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/rent-price-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"property_type": "Apartment", "neighborhood": "Kilimani",
#             "furnishing_status": "Semi-Furnished", "size_sqm": 95,
#             "num_bedrooms": 3, "num_bathrooms": 2, "distance_to_cbd_km": 5.5,
#             "service_charge_kes": 15000, "garage_size_sqm": 0,
#             "security_rating": 4, "amenities_score": 60,
#             "nearby_schools_count": 3, "has_garage": 0}'
#
# Sample request (PowerShell):
#   Invoke-RestMethod -Method Post `
#     -Uri "http://127.0.0.1:5000/api/v1/models/rent-price-regressor/predictions" `
#     -ContentType "application/json" `
#     -Body (@{property_type="Apartment"; neighborhood="Kilimani";
#              furnishing_status="Semi-Furnished"; size_sqm=95;
#              num_bedrooms=3; num_bathrooms=2; distance_to_cbd_km=5.5;
#              service_charge_kes=15000; garage_size_sqm=0; security_rating=4;
#              amenities_score=60; nearby_schools_count=3; has_garage=0} | ConvertTo-Json)

@app.route("/api/v1/models/rent-price-regressor/predictions", methods=["POST"])
def predict_rent():
    payload = get_json_body()

    input_row = build_input_row(payload, RENT_CATEGORICAL, RENT_NUMERIC)

    predicted_rent_kes = float(regression_model.predict(input_row)[0])

    return jsonify({
        "predicted_monthly_rent_kes": round(predicted_rent_kes, 2),
        "model": "Ridge Regression (log1p target)",
    })


# 2. CLASSIFICATION -- predict whether an employee is likely to leave (Yes/No)

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/employee-attrition-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"department": "Sales", "job_role_level": "Junior",
#             "education_level": "Bachelor'"'"'s", "performance_rating": "Average",
#             "years_since_last_promotion": 4, "monthly_salary_kes": 65000,
#             "overtime_hours_per_month": 30, "job_satisfaction_score": 2,
#             "work_life_balance_score": 2, "distance_from_home_km": 18,
#             "num_companies_worked_previously": 4,
#             "remote_work_days_per_week": 0, "bonus_amount_kes": null}'

@app.route("/api/v1/models/employee-attrition-classifier/predictions", methods=["POST"])
def predict_attrition():
    payload = get_json_body()
    input_row = build_input_row(payload, ATTRITION_CATEGORICAL, ATTRITION_NUMERIC)

    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_
    class_probabilities = {
        str(label): round(float(prob), 4)
        for label, prob in zip(class_labels, predicted_probabilities)
    }

    return jsonify({
        "predicted_attrition": str(predicted_class),
        "attrition_probability": class_probabilities.get("Yes"),
        "class_probabilities": class_probabilities,
        "model": "Logistic Regression (SMOTE-resampled at training only)",
    })


# 3. CLUSTERING -- assign a retail store to an existing segment

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/retail-store-segmenter/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"monthly_revenue_kes": 1200000, "daily_foot_traffic": 400,
#             "staff_count": 9, "store_size_sqm": 300,
#             "avg_basket_size_kes": 120}'

@app.route("/api/v1/models/retail-store-segmenter/predictions", methods=["POST"])
def predict_store_segment():
    payload = get_json_body()
    input_row = build_input_row(payload, [], STORE_NUMERIC)

    imputed = pd.DataFrame(clustering_imputer.transform(input_row), columns=STORE_NUMERIC)
    scaled_input = clustering_scaler.transform(imputed)
    assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "model": f"K-Means (k={clustering_kmeans.n_clusters}, imputed + standardized features)",
    })


# 4. ASSOCIATION RULE MINING -- recommend tracks for a listening session

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/music-session-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"session": ["M01", "M03"]}'
#
#   curl -X POST http://127.0.0.1:5000/api/v1/models/music-session-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"session": ["M08", "M10"], "top_n": 3}'

@app.route("/api/v1/models/music-session-recommender/predictions", methods=["POST"])
def recommend_tracks():
    payload = get_json_body()

    session = payload.get("session", payload.get("cart"))
    if not isinstance(session, list) or len(session) == 0:
        raise ValueError('"session" must be a non-empty list of track ids, e.g. ["M01", "M03"].')
    session = {str(track).strip().upper() for track in session}
    try:
        top_n = int(payload.get("top_n", 3))
    except (TypeError, ValueError):
        raise ValueError('"top_n" must be an integer.')

    unknown_tracks = sorted(t for t in session if t not in track_lookup.index)

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(session):
            for track_id, confidence, lift in consequent_list:
                if track_id not in session:
                    if track_id not in candidates or confidence > candidates[track_id][0]:
                        candidates[track_id] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "track_id": str(track_id),
            "item_name": track_lookup.loc[track_id, "item_name"],
            "category": track_lookup.loc[track_id, "category"],
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for track_id, (confidence, lift) in ranked
    ]

    return jsonify({
        "session": sorted(session),
        "unknown_tracks": unknown_tracks,
        "recommendations": recommendations,
        "model": "Apriori association rules, cleaned + precomputed lookup",
    })


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

